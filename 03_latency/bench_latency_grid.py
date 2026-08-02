"""
Unified per-tokenizer agent-latency grid -- ONE internally-consistent source for
the paper's Table 2 (latency), consolidating the r50k-only pieces previously
spread across 03_latency/bench_agent_mode_v2.py (read components),
03_latency/bench_agent_writer.py (write components),
07_kalcher_baseline/bench_unified_latency.py (r50k read+write) and
10_code_dict/bench_code_dict.py (+dict read).

For every token-ID compression method, BOTH agent directions, across all three
tiktoken tokenizers (r50k/cl100k/o200k) and all three domains:

  agent WRITE = from token IDs -> stored bytes (encode; incl. codec compress).
  agent READ  = from stored bytes -> token IDs (decode; incl. codec decode; +
                LEB128 + rank un-permute for +freq / Kalcher / dict-freqvarint).

Token methods:  raw pack, +freq (streamvbyte), +ANS, +dict (zstd-22 with a 112K
corpus-trained dictionary applied DIRECTLY to the raw packed token-ID bytes,
uint16/3-byte -- the token-domain parallel of byte zstd --train; no freq-remap,
no varint; read = decompress + unpack), Kalcher(zstd-22), Kalcher(LZMA), and
dict-freqvarint (the OLD +dict pipeline: freq-remap + LEB128 varint + dict, kept
only as a comparison row). The +dict WRITE latency (pack + dict-compress) was
not timed anywhere before -- it is measured here.

Byte-codec reference (tokenizer-independent, measured once per domain on the
r50k chunk sample): LZ4 / gzip-9 / zstd-19 / brotli-q11 / zstd --train, reported
as compress_only + decompress_only components (as 03_latency now does). The byte
path's mandatory tokenize/detokenize tax is reported per tokenizer as
SERVING-COLD single shots (timed_once_serving): the CPU cache is swept with a
64 MB buffer before each shot, so the tokenizer's multi-MB rank table is cold, as
it is in real serving where interleaved codec and model work evicts it. This is
the same protocol as 09_cold_tokenize, so `tokenize_serving_cold_us` is now the
serving-cold cost (English: r50k 240us, cl100k 306us, o200k 281us) instead of the
~97-145us text-cold figure a back-to-back loop produces. 09_cold_tokenize's
dedicated bench independently gets 248/291/267us on the same protocol.

RUN PINNED: `taskset -c 4 env RAYON_NUM_THREADS=1 ... uv run python <this>`. This
is a single-core bench on a hybrid CPU (P-cores 4.8GHz, LP-E 2.5GHz); an unpinned
run can migrate onto an LP-E core and inflate every cell ~1.6x.

Conventions: codec ops = timed_reps (warm, median-of-30); tokenize/detokenize =
timed_once_serving (cache-evicted single shot); per-cell deterministic seeds
default_rng([SEED, tok_idx, domain_idx]); full-train rank/ANS tables; ratios and
round-trip asserts kept. English (prose) is the headline; all domains computed.
"""
import os
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TIKTOKEN_MAX_THREADS", "1")

import sys
import json
import gzip
import lzma
import numpy as np
import tiktoken
import lz4.frame as lz4f
import zstandard as zstd
import brotli
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import (
    load_ids, make_chunks, bootstrap_ci, timed_reps, timed_once, timed_once_serving, seeded_rng,
    pack3, unpack3, leb128_encode, leb128_decode, svb_encode_arr, svb_decode_arr,
    build_rank_table, build_ans_model, full_train_zstd_dict, zstd_c22, zstd_d, LZMA_FILTERS,
)

DOMAINS = ["prose", "code", "hindi"]
TOKENIZERS = {"r50k": ("r50k_base", 50257), "cl100k": ("cl100k_base", 100277), "o200k": ("o200k_base", 200019)}
CHUNK_SIZE = 512
N_CHUNKS = 40
SEED = 9012                     # match Table 1 path B / 10_code_dict
DICT_SIZE = 112 * 1024          # token-ID zstd --train dict
DICT_TRAIN_SAMPLES = 8000       # bounded pool of train windows for the dict
# dict-freqvarint (old freq-remap+LEB128+dict comparison) is appended LAST so its
# bootstrap draws don't perturb the RNG order of the other methods.
TOKEN_METHODS = ["raw", "+freq", "+ANS", "+dict", "Kalcher(zstd)", "Kalcher(LZMA)", "dict-freqvarint"]
BYTE_CODECS = ["LZ4", "gzip-9", "zstd-19", "brotli-q11", "zstd --train"]

r50k = tiktoken.get_encoding("r50k_base")


def _mk_dict(samples):
    zdict = zstd.ZstdCompressionDict(zstd.train_dictionary(DICT_SIZE, samples).as_bytes())
    return zstd.ZstdCompressor(level=22, dict_data=zdict), zstd.ZstdDecompressor(dict_data=zdict)


def build_token_dicts(train, enc, rank_of, is_r50k, fits16, rng):
    """Two 112K zstd dicts trained corpus-wide on the same TRAIN 512-token windows
    (one rng.choice, so chunk selection downstream is unchanged):
      idbytes -> +dict:          raw packed token-ID bytes (uint16 / 3-byte).
      varint  -> dict-freqvarint: freq-remapped LEB128 varint bytes (old pipeline).
    Returns ((zc_idb, zd_idb), (zc_var, zd_var))."""
    n_windows = len(train) // CHUNK_SIZE
    idxs = np.arange(n_windows)
    if n_windows > DICT_TRAIN_SAMPLES:
        idxs = rng.choice(n_windows, size=DICT_TRAIN_SAMPLES, replace=False)
    s_idb, s_var = [], []
    for i in idxs:
        w = train[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        ids = w if is_r50k else np.array(enc.encode(r50k.decode(w.tolist()), disallowed_special=()), dtype=np.int64)
        s_idb.append(ids.astype(np.uint16).tobytes() if fits16 else pack3(ids))
        s_var.append(leb128_encode(rank_of[ids]))
    return _mk_dict(s_idb), _mk_dict(s_var)


def byte_codec_components(texts, train, rng):
    """compress_only + decompress_only (warm) and ratio for each byte codec, on
    the given test texts. Tokenizer-independent."""
    raw_byte_lists = [t.encode("utf-8") for t in texts]
    # zstd --train: the shared FULL-train-split recipe from tnbench (same one Table 1
    # uses), NOT a 400-window subset. On code the full split reaches ~3.47x vs ~2.80x
    # from 400 windows, so this keeps the sweep's zstd --train consistent with Table 1.
    zc19, zd19 = zstd.ZstdCompressor(level=19), zstd.ZstdDecompressor()
    zcd, zdd = full_train_zstd_dict(train, r50k)
    codecs = {
        "LZ4": (lz4f.compress, lz4f.decompress),
        "gzip-9": (lambda b: gzip.compress(b, 9), gzip.decompress),
        "zstd-19": (zc19.compress, zd19.decompress),
        "brotli-q11": (lambda b: brotli.compress(b, quality=11), brotli.decompress),
        "zstd --train": (zcd.compress, zdd.decompress),
    }
    out = {}
    # Iterate the canonical BYTE_CODECS list (single source of truth), not codecs.items(),
    # so the byte-codec set can never silently diverge from Table 1 / the chunk-sweep: a
    # canonical codec missing an implementation here raises KeyError instead of vanishing.
    for name in BYTE_CODECS:
        comp, decomp = codecs[name]
        ratios, ct, dt = [], [], []
        for raw in raw_byte_lists:
            ct.append(timed_reps(lambda r=raw: comp(r)))
            c = comp(raw)
            assert decomp(c) == raw
            ratios.append(len(raw) / len(c))
            dt.append(timed_reps(lambda c=c: decomp(c)))
        out[name] = {
            "compress_us": bootstrap_ci(ct, rng),
            "decompress_us": bootstrap_ci(dt, rng),
            "ratio": bootstrap_ci(ratios, rng),
        }
    return out


def run_cell(domain, tok, enc, vocab, is_r50k, rng):
    train = load_ids(f"{domain}_train")
    test = load_ids(f"{domain}_test")
    train_ids = train if is_r50k else np.array(enc.encode(r50k.decode(train.tolist()), disallowed_special=()), dtype=np.int64)
    rank_of, token_of_rank = build_rank_table(train_ids, vocab)
    ans_model = build_ans_model(train_ids, vocab)
    fits16 = vocab <= 65536
    (zc_idb, zd_idb), (zc_var, zd_var) = build_token_dicts(train, enc, rank_of, is_r50k, fits16, rng)

    def pack_ids(i):
        return i.astype(np.uint16).tobytes() if fits16 else pack3(i)

    def unpack_ids(buf, n):
        return np.frombuffer(buf, dtype=np.uint16).astype(np.int64) if fits16 else unpack3(buf, n)

    chunks = make_chunks(test, CHUNK_SIZE, N_CHUNKS, rng)
    texts = [r50k.decode(c.tolist()) for c in chunks]

    write = {m: [] for m in TOKEN_METHODS}
    read = {m: [] for m in TOKEN_METHODS}
    ratio = {m: [] for m in TOKEN_METHODS}
    tok_cold, detok_cold = [], []

    # Serving-cold tokenize / detokenize (the byte path's mandatory tax) runs in
    # its OWN pass, before the codec loop. timed_once_serving sweeps 64 MB to
    # evict the tokenizer's multi-MB rank table, as real serving does, which is
    # why this reports ~240us for r50k where a plain timed_once reported ~125us
    # (the back-to-back loop kept the table resident). Keeping it in a separate
    # pass means the sweep does not pollute the codec timings below.
    # `.tolist()` is hoisted OUT of the timed lambda: it is numpy-to-list marshalling
    # this harness happens to need, not part of detokenize, and charging it inflated
    # detokenize ~20% (55us vs 09_cold_tokenize's 45us for r50k) on the same protocol.
    for text in texts:
        ids_list = enc.encode(text, disallowed_special=())
        tok_cold.append(timed_once_serving(lambda t=text: enc.encode(t, disallowed_special=())))
        detok_cold.append(timed_once_serving(lambda i=ids_list: enc.decode(i)))

    for text in texts:
        ids = np.array(enc.encode(text, disallowed_special=()), dtype=np.int64)
        n = len(ids)
        raw_len = len(text.encode("utf-8"))
        remapped = rank_of[ids]
        varint = leb128_encode(remapped)

        # ---- raw pack ----
        packed = ids.astype(np.uint16).tobytes() if fits16 else pack3(ids)
        if fits16:
            write["raw"].append(timed_reps(lambda i=ids: i.astype(np.uint16).tobytes()))
            read["raw"].append(timed_reps(lambda p=packed: np.frombuffer(p, dtype=np.uint16)))
        else:
            write["raw"].append(timed_reps(lambda i=ids: pack3(i)))
            read["raw"].append(timed_reps(lambda p=packed, n=n: unpack3(p, n)))
        ratio["raw"].append(raw_len / len(packed))

        # ---- +freq (streamvbyte) ----
        svbp = svb_encode_arr(remapped)
        assert np.array_equal(token_of_rank[svb_decode_arr(svbp, n)], ids)
        write["+freq"].append(timed_reps(lambda i=ids: svb_encode_arr(rank_of[i])))
        read["+freq"].append(timed_reps(lambda p=svbp, n=n: token_of_rank[svb_decode_arr(p, n)]))
        ratio["+freq"].append(raw_len / len(svbp))

        # ---- +ANS ----
        def ans_write(i32=ids.astype(np.int32)):
            c = constriction.stream.stack.AnsCoder()
            c.encode_reverse(i32, ans_model)
            return c.get_compressed().tobytes()

        ansp = ans_write()
        ans_arr = np.frombuffer(ansp, dtype=np.uint32).copy()

        def ans_read(a=ans_arr, n=n):
            return constriction.stream.stack.AnsCoder(a.copy()).decode(ans_model, n)

        assert np.array_equal(np.asarray(ans_read()), ids.astype(np.int32))
        write["+ANS"].append(timed_reps(ans_write))
        read["+ANS"].append(timed_reps(ans_read))
        ratio["+ANS"].append(raw_len / len(ansp))

        # ---- +dict (paper): zstd-22 dict applied DIRECTLY to raw packed token-ID
        #      bytes (uint16 / 3-byte). No freq-remap, no varint. write = pack +
        #      dict-compress; read = dict-decompress + unpack. ----
        packed = pack_ids(ids)
        dictp = zc_idb.compress(packed)
        assert np.array_equal(unpack_ids(zd_idb.decompress(dictp), n), ids)
        write["+dict"].append(timed_reps(lambda i=ids: zc_idb.compress(pack_ids(i))))
        read["+dict"].append(timed_reps(lambda p=dictp, n=n: unpack_ids(zd_idb.decompress(p), n)))
        ratio["+dict"].append(raw_len / len(dictp))

        # ---- Kalcher(zstd-22, no dict) ----
        kzp = zstd_c22.compress(varint)
        write["Kalcher(zstd)"].append(timed_reps(lambda i=ids: zstd_c22.compress(leb128_encode(rank_of[i]))))
        read["Kalcher(zstd)"].append(timed_reps(lambda p=kzp: token_of_rank[leb128_decode(zstd_d.decompress(p))]))
        ratio["Kalcher(zstd)"].append(raw_len / len(kzp))

        # ---- Kalcher(LZMA) ----
        klp = lzma.compress(varint, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
        write["Kalcher(LZMA)"].append(timed_reps(
            lambda i=ids: lzma.compress(leb128_encode(rank_of[i]), format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)))
        read["Kalcher(LZMA)"].append(timed_reps(
            lambda p=klp: token_of_rank[leb128_decode(lzma.decompress(p, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS))]))
        ratio["Kalcher(LZMA)"].append(raw_len / len(klp))

        # ---- dict-freqvarint (comparison): old freq-remap + LEB128 + dict pipeline ----
        fvp = zc_var.compress(varint)
        assert np.array_equal(token_of_rank[leb128_decode(zd_var.decompress(fvp))], ids)
        write["dict-freqvarint"].append(timed_reps(lambda i=ids: zc_var.compress(leb128_encode(rank_of[i]))))
        read["dict-freqvarint"].append(timed_reps(lambda p=fvp: token_of_rank[leb128_decode(zd_var.decompress(p))]))
        ratio["dict-freqvarint"].append(raw_len / len(fvp))

    methods = {m: {"write_us": bootstrap_ci(write[m], rng),
                   "read_us": bootstrap_ci(read[m], rng),
                   "ratio": bootstrap_ci(ratio[m], rng)} for m in TOKEN_METHODS}
    return {
        "methods": methods,
        "tokenize_serving_cold_us": bootstrap_ci(tok_cold, rng),
        "detokenize_serving_cold_us": bootstrap_ci(detok_cold, rng),
    }, texts, train


def chunk_sweep():
    """Chunk-size scaling appendix: the FULL cross product of every tokenizer x every
    corpus x every chunk size x every method, reusing run_cell + byte_codec_components
    (no bench logic duplicated). Reports agent READ and WRITE latency (paper
    definition) and ratio, plus a per-1k-token normalized view. Reassigns the
    module-global CHUNK_SIZE, which every helper reads.

    Every tokenizer is run on every corpus (not just each corpus's native one) so the
    reader can see that token-native beats byte codecs across ALL tokenizers, and that
    a script-mismatched tokenizer (e.g. r50k on Hindi, no Devanagari merges) is what
    understates token-native, not the method. The tokenizer dimension is a loop over
    TOKENIZERS, not a hardcoded choice, so this generalizes.

    Byte-codec ratio and compress/decompress are tokenizer-independent, so they are
    measured ONCE per (corpus, chunk_size) and reused across tokenizers; but the agent
    read/write of a byte store still pays the tokenize/detokenize of whichever
    tokenizer the model uses, so byte read/write ARE reported per tokenizer (byte
    codec decompress/compress + that tokenizer's serving-cold (de)tokenize). The chunk
    sample is seeded per (chunk_size, corpus) and is identical across tokenizers, so
    the once-measured byte codec numbers line up with every tokenizer's token rows."""
    global CHUNK_SIZE
    CHUNK_SIZES = [512, 1024, 2048, 4096]
    TOK_SHOW = ["raw", "+freq", "+ANS", "+dict"]
    # Draw byte codecs from the canonical module-level BYTE_CODECS (LZ4, gzip-9, zstd-19,
    # brotli-q11, zstd --train) so the sweep can never again silently drop zstd --train or
    # gzip-9 relative to Table 1. zstd --train / gzip-9 are byte codecs, so like the others
    # they are tokenizer-independent and measured once per (corpus, chunk_size).
    BYTE_SHOW = list(BYTE_CODECS)
    ALLM = BYTE_SHOW + TOK_SHOW
    enc_cache = {}
    sweep = {}  # f"{tok}|{domain}|{cs}" -> {tokenize_cold_us, detokenize_cold_us, methods}
    for cs in CHUNK_SIZES:
        CHUNK_SIZE = cs
        for di, domain in enumerate(DOMAINS):
            byte = None  # measured once per (domain, cs), reused across tokenizers
            for tok, (enc_name, vocab) in TOKENIZERS.items():
                enc = enc_cache.setdefault(tok, tiktoken.get_encoding(enc_name))
                print(f"########## cs={cs} / {domain} / {tok} ##########", flush=True)
                rng = seeded_rng(SEED, cs, di)  # SAME across tokenizers -> same chunks
                cell, texts, train = run_cell(domain, tok, enc, vocab, tok == "r50k", rng)
                if byte is None:
                    byte = byte_codec_components(texts, train, rng)
                tok_cold = cell["tokenize_serving_cold_us"][0]
                detok_cold = cell["detokenize_serving_cold_us"][0]
                rec = {"tokenize_cold_us": tok_cold, "detokenize_cold_us": detok_cold, "methods": {}}
                for m in TOK_SHOW:
                    mm = cell["methods"][m]
                    rec["methods"][m] = {"kind": "token", "read_us": mm["read_us"][0],
                                         "write_us": mm["write_us"][0], "ratio": mm["ratio"][0]}
                for name in BYTE_SHOW:
                    b = byte[name]
                    # agent read = decompress + tokenize (bytes -> IDs);
                    # agent write = compress + detokenize (IDs -> stored bytes).
                    # Byte ratio/compress/decompress are tokenizer-independent; only the
                    # tokenize/detokenize tax varies with tok.
                    rec["methods"][name] = {"kind": "byte",
                                            "read_us": b["decompress_us"][0] + tok_cold,
                                            "write_us": b["compress_us"][0] + detok_cold,
                                            "compress_us": b["compress_us"][0],
                                            "decompress_us": b["decompress_us"][0],
                                            "ratio": b["ratio"][0]}
                sweep[f"{tok}|{domain}|{cs}"] = rec

    def sub_tables(title, field, fmt, norm=False):
        print(f"\n{'#' * 96}\n  {title}\n{'#' * 96}")
        for tok in TOKENIZERS:
            print(f"\n  --- tokenizer = {tok} ---  (cells = prose / code / hindi{', us/1k-tok' if norm else ''})")
            print(f"  {'method':<12}{'kind':<6}" + "".join(f"{str(cs):>21}" for cs in CHUNK_SIZES))
            for m in ALLM:
                kind = sweep[f"{tok}|prose|512"]["methods"][m]["kind"]
                row = f"  {m:<12}{kind:<6}"
                for cs in CHUNK_SIZES:
                    cells = []
                    for d in DOMAINS:
                        v = sweep[f"{tok}|{d}|{cs}"]["methods"][m][field]
                        if norm:
                            v = v / cs * 1000
                        cells.append(fmt.format(v))
                    row += ("/".join(cells)).rjust(21)
                print(row)

    print(f"\nFULL cross product: tokenizer x corpus x chunk size x method, seed {SEED}")
    sub_tables("RATIO (median x vs raw UTF-8)", "ratio", "{:.2f}")
    sub_tables("AGENT READ latency us (byte = decompress + tokenize serving-cold)", "read_us", "{:.0f}")
    sub_tables("AGENT WRITE latency us (byte = compress + detokenize serving-cold)", "write_us", "{:.0f}")
    sub_tables("AGENT READ latency per 1k tokens (linearity)", "read_us", "{:.0f}", norm=True)
    sub_tables("AGENT WRITE latency per 1k tokens (linearity)", "write_us", "{:.0f}", norm=True)

    print(f"\n  tokenize / detokenize serving-cold (us), per tokenizer, by chunk size (prose/code/hindi):")
    for tok in TOKENIZERS:
        for cs in CHUNK_SIZES:
            tks = "/".join(f"{sweep[f'{tok}|{d}|{cs}']['tokenize_cold_us']:.0f}" for d in DOMAINS)
            dtk = "/".join(f"{sweep[f'{tok}|{d}|{cs}']['detokenize_cold_us']:.0f}" for d in DOMAINS)
            print(f"    {tok:<7} cs={cs:<5} tokenize={tks:<26} detokenize={dtk}")

    out = {
        "config": {"tokenizers": list(TOKENIZERS), "domains": DOMAINS, "chunk_sizes": CHUNK_SIZES,
                   "n_chunks": N_CHUNKS, "seed": SEED, "token_methods": TOK_SHOW, "byte_codecs": BYTE_SHOW,
                   "key": "sweep[f'{tokenizer}|{corpus}|{chunk_size}'] -> {methods: {method: {ratio, read_us, write_us}}}",
                   "read": "agent read = to token IDs; byte = decompress + tokenize(serving-cold)",
                   "write": "agent write = from token IDs; byte = compress + detokenize(serving-cold)",
                   "byte_note": "byte ratio/compress/decompress are tokenizer-independent (measured once per corpus,cs); "
                                "byte read/write vary by tokenizer only through the (de)tokenize tax",
                   "timing": "codec ops warm median-of-30 (timed_reps); tokenize/detokenize serving-cold (timed_once_serving, 64MB sweep)"},
        "sweep": sweep,
    }
    out_path = os.path.join(os.path.dirname(__file__), "latency_chunksize_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


def main():
    grid = {d: {} for d in DOMAINS}
    byte_ref = {}
    for di, domain in enumerate(DOMAINS):
        for ti, (tok, (enc_name, vocab)) in enumerate(TOKENIZERS.items()):
            print(f"########## {domain} / {tok} ##########", flush=True)
            enc = tiktoken.get_encoding(enc_name)
            rng = seeded_rng(SEED, ti, di)
            cell, texts, train = run_cell(domain, tok, enc, vocab, tok == "r50k", rng)
            grid[domain][tok] = cell
            m = cell["methods"]
            for name in TOKEN_METHODS:
                print(f"  {name:<14} write={m[name]['write_us'][0]:8.1f}us  read={m[name]['read_us'][0]:8.1f}us  ratio={m[name]['ratio'][0]:.2f}x", flush=True)
            # byte codecs once per domain, on the r50k chunk sample
            if tok == "r50k":
                byte_ref[domain] = byte_codec_components(texts, train, rng)

    # ── English (prose) headline grid ──
    print(f"\n{'=' * 100}")
    print("TABLE 2 (unified): agent latency us/chunk -- English (C4), 512-tok, per tokenizer [write | read]")
    print(f"{'=' * 100}")
    hdr = "  " + f"{'method':<15}" + "".join(f"{t:>26}" for t in TOKENIZERS)
    print(hdr)
    for name in TOKEN_METHODS:
        row = f"  {name:<15}"
        for tok in TOKENIZERS:
            w = grid["prose"][tok]["methods"][name]["write_us"][0]
            r = grid["prose"][tok]["methods"][name]["read_us"][0]
            row += f"{w:>11.1f} |{r:>11.1f} ".rjust(26)
        print(row)
    print("  " + "-" * 90)
    for tok in TOKENIZERS:
        c = grid["prose"][tok]
        print(f"  [{tok}] tokenize(serving-cold)={c['tokenize_serving_cold_us'][0]:.1f}us  "
              f"detokenize={c['detokenize_serving_cold_us'][0]:.1f}us")
    print(f"\n  byte-codec reference (English, compress|decompress us):")
    for name in BYTE_CODECS:
        b = byte_ref["prose"][name]
        print(f"    {name:<14} compress={b['compress_us'][0]:8.1f}us  decompress={b['decompress_us'][0]:7.1f}us  ratio={b['ratio'][0]:.2f}x")

    out = {
        "config": {
            "chunk_size": CHUNK_SIZE, "n_chunks": N_CHUNKS, "seed": SEED,
            "tokenizers": list(TOKENIZERS), "domains": DOMAINS,
            "token_methods": TOKEN_METHODS, "byte_codecs": BYTE_CODECS,
            "dict": f"+dict = zstd-22, {DICT_SIZE // 1024}K, corpus-trained on RAW PACKED token-ID "
                    "bytes (uint16/3-byte); dict-freqvarint = old comparison (freq-remap+LEB128 varint)",
            "seed_scheme": "per-cell default_rng([SEED, tok_idx, domain_idx])",
            "timing": "codec ops warm median-of-30 (timed_reps). tokenize/detokenize serving-cold single shot (timed_once_serving): 64MB cache sweep before each shot so the rank table is evicted, same protocol as 09_cold_tokenize",
            "read": "to token IDs incl. codec decode (+dict: unpack; +freq/Kalcher/dict-freqvarint: LEB128 + rank un-permute)",
            "write": "from token IDs incl. codec compress (+dict: pack + compress)",
        },
        "grid": grid,
        "byte_codecs": byte_ref,
    }
    out_path = os.path.join(os.path.dirname(__file__), "latency_grid_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    if "--chunk-sweep" in sys.argv:
        chunk_sweep()
    else:
        main()
