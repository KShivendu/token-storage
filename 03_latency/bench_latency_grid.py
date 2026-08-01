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
  agent READ  = from stored bytes -> token IDs (decode; incl. codec decode +
                LEB128 + rank un-permute for +freq / +dict / Kalcher).

Token methods:  raw pack, +freq (streamvbyte), +ANS, +dict (zstd-22 with a 112K
corpus-trained dictionary over freq-remapped LEB128 varints), Kalcher(zstd-22),
Kalcher(LZMA). The +dict WRITE latency (zstd-22-with-dict compress of the varint)
was not timed anywhere before -- it is measured here.

Byte-codec reference (tokenizer-independent, measured once per domain on the
r50k chunk sample): LZ4 / gzip-9 / zstd-19 / brotli-q11 / zstd --train, reported
as compress_only + decompress_only components (as 03_latency now does). The byte
path's mandatory tokenize/detokenize tax is reported per tokenizer as
serving-cold single shots.

Conventions: codec ops = timed_reps (warm, median-of-30); tokenize/detokenize =
timed_once (serving-cold, single shot); per-cell deterministic seeds
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
    load_ids, make_chunks, bootstrap_ci, timed_reps, timed_once, seeded_rng,
    pack3, unpack3, leb128_encode, leb128_decode, svb_encode_arr, svb_decode_arr,
    build_rank_table, build_ans_model, zstd_c22, zstd_d, LZMA_FILTERS,
)

DOMAINS = ["prose", "code", "hindi"]
TOKENIZERS = {"r50k": ("r50k_base", 50257), "cl100k": ("cl100k_base", 100277), "o200k": ("o200k_base", 200019)}
CHUNK_SIZE = 512
N_CHUNKS = 40
SEED = 9012                     # match Table 1 path B / 10_code_dict
DICT_SIZE = 112 * 1024          # token-ID zstd --train dict
DICT_TRAIN_SAMPLES = 8000       # bounded pool of train windows for the dict
TOKEN_METHODS = ["raw", "+freq", "+ANS", "+dict", "Kalcher(zstd)", "Kalcher(LZMA)"]
BYTE_CODECS = ["LZ4", "gzip-9", "zstd-19", "brotli-q11", "zstd --train"]

r50k = tiktoken.get_encoding("r50k_base")


def build_token_dict(train, enc, rank_of, is_r50k, rng):
    """112K zstd dict trained corpus-wide on freq-remapped LEB128 varints of the
    train split's 512-token windows (same object shape as a stored payload)."""
    n_windows = len(train) // CHUNK_SIZE
    idxs = np.arange(n_windows)
    if n_windows > DICT_TRAIN_SAMPLES:
        idxs = rng.choice(n_windows, size=DICT_TRAIN_SAMPLES, replace=False)
    samples = []
    for i in idxs:
        w = train[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        ids = w if is_r50k else np.array(enc.encode(r50k.decode(w.tolist()), disallowed_special=()), dtype=np.int64)
        samples.append(leb128_encode(rank_of[ids]))
    zdict = zstd.ZstdCompressionDict(zstd.train_dictionary(DICT_SIZE, samples).as_bytes())
    return zstd.ZstdCompressor(level=22, dict_data=zdict), zstd.ZstdDecompressor(dict_data=zdict)


def byte_codec_components(texts, train, rng):
    """compress_only + decompress_only (warm) and ratio for each byte codec, on
    the given test texts. Tokenizer-independent."""
    raw_byte_lists = [t.encode("utf-8") for t in texts]
    # zstd --train byte dict on this domain's train utf-8 windows (matches 03)
    train_chunks_bytes = [
        r50k.decode(train[i:i + CHUNK_SIZE].tolist()).encode("utf-8")
        for i in range(0, min(len(train), CHUNK_SIZE * 400), CHUNK_SIZE)
    ]
    zdict = zstd.ZstdCompressionDict(zstd.train_dictionary(112 * 1024, train_chunks_bytes).as_bytes())
    zdict.precompute_compress(level=19)
    zc19, zd19 = zstd.ZstdCompressor(level=19), zstd.ZstdDecompressor()
    zcd, zdd = zstd.ZstdCompressor(level=19, dict_data=zdict), zstd.ZstdDecompressor(dict_data=zdict)
    codecs = {
        "LZ4": (lz4f.compress, lz4f.decompress),
        "gzip-9": (lambda b: gzip.compress(b, 9), gzip.decompress),
        "zstd-19": (zc19.compress, zd19.decompress),
        "brotli-q11": (lambda b: brotli.compress(b, quality=11), brotli.decompress),
        "zstd --train": (zcd.compress, zdd.decompress),
    }
    out = {}
    for name, (comp, decomp) in codecs.items():
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
    zc_dict, zd_dict = build_token_dict(train, enc, rank_of, is_r50k, rng)
    fits16 = vocab <= 65536

    chunks = make_chunks(test, CHUNK_SIZE, N_CHUNKS, rng)
    texts = [r50k.decode(c.tolist()) for c in chunks]

    write = {m: [] for m in TOKEN_METHODS}
    read = {m: [] for m in TOKEN_METHODS}
    ratio = {m: [] for m in TOKEN_METHODS}
    tok_cold, detok_cold = [], []

    for text in texts:
        ids = np.array(enc.encode(text, disallowed_special=()), dtype=np.int64)
        n = len(ids)
        raw_len = len(text.encode("utf-8"))
        remapped = rank_of[ids]
        varint = leb128_encode(remapped)

        # serving-cold tokenize / detokenize (the byte path's mandatory tax)
        tok_cold.append(timed_once(lambda t=text: enc.encode(t, disallowed_special=())))
        detok_cold.append(timed_once(lambda i=ids: enc.decode(i.tolist())))

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

        # ---- +dict (zstd-22 with corpus-trained token-ID dict) ----
        dictp = zc_dict.compress(varint)
        assert np.array_equal(token_of_rank[leb128_decode(zd_dict.decompress(dictp))], ids)
        write["+dict"].append(timed_reps(lambda i=ids: zc_dict.compress(leb128_encode(rank_of[i]))))
        read["+dict"].append(timed_reps(lambda p=dictp: token_of_rank[leb128_decode(zd_dict.decompress(p))]))
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

    methods = {m: {"write_us": bootstrap_ci(write[m], rng),
                   "read_us": bootstrap_ci(read[m], rng),
                   "ratio": bootstrap_ci(ratio[m], rng)} for m in TOKEN_METHODS}
    return {
        "methods": methods,
        "tokenize_serving_cold_us": bootstrap_ci(tok_cold, rng),
        "detokenize_serving_cold_us": bootstrap_ci(detok_cold, rng),
    }, texts, train


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
            "dict": f"zstd-22, {DICT_SIZE // 1024}K, corpus-trained on freq-remapped LEB128 varints",
            "seed_scheme": "per-cell default_rng([SEED, tok_idx, domain_idx])",
            "timing": "codec ops warm median-of-30 (timed_reps); tokenize/detokenize serving-cold single shot (timed_once)",
            "read": "to token IDs incl. codec decode + LEB128 + rank un-permute",
            "write": "from token IDs incl. codec compress",
        },
        "grid": grid,
        "byte_codecs": byte_ref,
    }
    out_path = os.path.join(os.path.dirname(__file__), "latency_grid_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
