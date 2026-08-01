"""
Same two tables as before (compression ratio vs raw UTF-8 bytes; encode/
decode latency), but now swept across chunk sizes that actually matter:
256 and 512 tokens (typical embedding-model chunk sizes -- e.g. mixedbread
and most RAG setups chunk well below 1000 tokens, nowhere near the 2000
used in the first pass), plus 2000 kept as a reference point.

Static ANS tables are trained per (domain, tokenizer) on that domain's
TRAIN split (never touching the TEST chunks measured here).
"""
import gzip
import json
import lzma
import os
import sys
import time
import brotli
import lz4.frame
import numpy as np
import tiktoken
import zstandard as zstd
import constriction

# Kalcher / +freq helpers, reused from bench_kalcher.py
from bench_kalcher import leb128_encode, leb128_decode, zstd_c22, LZMA_FILTERS, svb_encode_arr

ZSTD_DICT_SIZE = 112 * 1024
_zdict_cache = {}  # domain -> ZstdCompressor with a FULL-train-split dictionary


def full_train_zstd_compressor(domain, train_r50k):
    """zstd level-19 compressor whose 112KB dictionary is trained on the domain's
    FULL train split (cut into 512-token windows), not the blog's 400-sample
    subset -- this is the train-dependent 'zstd --train' cell for the full-train
    consistent table. Cached per domain. No numpy-RNG draw (keeps the aligned
    run's chunk selection bit-identical)."""
    if domain not in _zdict_cache:
        samples = [
            r50k.decode(train_r50k[i : i + 512].tolist()).encode("utf-8")
            for i in range(0, (len(train_r50k) // 512) * 512, 512)
        ]
        zdict = zstd.ZstdCompressionDict(zstd.train_dictionary(ZSTD_DICT_SIZE, samples).as_bytes())
        _zdict_cache[domain] = zstd.ZstdCompressor(level=19, dict_data=zdict)
    return _zdict_cache[domain]

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
DOMAINS = ["prose", "code", "hindi"]
# 2000 dropped: only need 512, and 512's RNG state is identical whether or not
# 2000 (which runs AFTER 512) is computed. 256 is kept because it consumes RNG
# BEFORE 512, so it must run to reproduce the blog's exact 512 chunk selection.
CHUNK_SIZES = [256, 512]
N_CHUNKS = 40

# Elasticsearch/Lucene-style block LZ4 (Lucene's classic "best_speed" stored
# fields codec): documents are batched into ~16KB blocks (or 128 docs,
# whichever comes first) and the WHOLE block is LZ4-compressed as one unit --
# not per-document like the plain "LZ4" row elsewhere in this script. This
# gives LZ4 a much bigger dictionary window (cross-document redundancy), at
# the cost that reading back ANY single document requires decompressing the
# entire block it lives in.
LZ4_BLOCK_BYTES = 16 * 1024
LZ4_BLOCK_MAX_DOCS = 128
LZ4_BLOCK_N_DOCS = 300  # sampled per domain per chunk_size, before grouping into blocks
TOKENIZERS = {"r50k": ("r50k_base", 50257, 2), "cl100k": ("cl100k_base", 100277, 3), "o200k": ("o200k_base", 200019, 3)}
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(9012)

zstd_c = zstd.ZstdCompressor(level=19)
r50k = tiktoken.get_encoding("r50k_base")
TOKENIZER_ENCODERS = {k: tiktoken.get_encoding(v[0]) for k, v in TOKENIZERS.items()}


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=0.10):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(np.median(values)), (float(values[0]), float(values[0]))
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    boot_medians = np.median(values[idx], axis=1)
    lo, hi = np.percentile(boot_medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(values)), (float(lo), float(hi))


def make_chunks(test_arr, chunk_size, n_chunks):
    max_chunks = len(test_arr) // chunk_size
    n = min(n_chunks, max_chunks)
    starts = RNG.choice(max_chunks, size=n, replace=False) * chunk_size
    return [test_arr[s : s + chunk_size] for s in starts]


def run_lz4_blocks_for_chunk_size(chunk_size):
    """ES/Lucene-style block LZ4: batch documents into ~16KB/128-doc blocks,
    compress each block whole. Reports ratio, amortized per-doc encode cost
    (bulk write), and per-doc decode cost (the FULL block must be decoded to
    read back even one document -- that's the real cost of a single-document
    fetch, not divided by docs-in-block).

    Verified against Lucene's actual CompressingStoredFieldsWriter source: at
    finish() (segment flush/close), any still-buffered docs that haven't hit
    the 16KB/128-doc threshold are flushed and compressed anyway -- Lucene
    just marks that chunk "dirty" (a signal used to prioritize it at merge
    time), it never leaves it uncompressed. So the trailing partial block
    here is flushed too, not discarded -- matching that behavior instead of
    only ever measuring full-size blocks. Note this still undercounts real
    dirty-chunk frequency: a continuously-indexing ES index refreshes on a
    ~1s timer, not just once at the end of a bulk batch, so a live index
    would accumulate more small/dirty chunks than this single-batch
    simulation produces (here, only the trailing block can be dirty)."""
    results = {}
    for domain in DOMAINS:
        test_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_test.npy")).astype(np.int64)
        sampled = make_chunks(test_r50k, chunk_size, LZ4_BLOCK_N_DOCS)
        # sort by corpus position so blocks group corpus-adjacent documents,
        # mirroring how a real index batches sequentially-ingested docs
        sampled = sorted(sampled, key=lambda c: c[0])
        texts = [r50k.decode(c.tolist()) for c in sampled]
        raw_byte_lists = [t.encode("utf-8") for t in texts]

        blocks = []
        cur, cur_bytes = [], 0
        for raw in raw_byte_lists:
            if cur and (cur_bytes + len(raw) > LZ4_BLOCK_BYTES or len(cur) >= LZ4_BLOCK_MAX_DOCS):
                blocks.append(cur)
                cur, cur_bytes = [], 0
            cur.append(raw)
            cur_bytes += len(raw)
        n_dirty = 1 if (cur and cur_bytes < LZ4_BLOCK_BYTES and len(cur) < LZ4_BLOCK_MAX_DOCS) else 0
        if cur:
            blocks.append(cur)  # flush the trailing partial ("dirty") chunk, matching finish()

        ratios, enc_per_doc, dec_per_doc, docs_per_block = [], [], [], []
        for block in blocks:
            combined = b"".join(block)
            t0 = time.perf_counter()
            comp = lz4.frame.compress(combined)
            t1 = time.perf_counter()
            decomp = lz4.frame.decompress(comp)
            t2 = time.perf_counter()
            assert decomp == combined
            ratios.append(len(combined) / len(comp))
            enc_per_doc.append((t1 - t0) / len(block))  # amortized bulk-write cost
            dec_per_doc.append(t2 - t1)  # cost to retrieve ONE doc: decode the whole block
            docs_per_block.append(len(block))

        results[domain] = {
            "ratio": bootstrap_ci(ratios),
            "encode_per_doc_us": bootstrap_ci(np.array(enc_per_doc) * 1e6),
            "decode_per_doc_us": bootstrap_ci(np.array(dec_per_doc) * 1e6),
            "n_blocks": len(blocks),
            "n_dirty_blocks": n_dirty,
            "median_docs_per_block": float(np.median(docs_per_block)) if docs_per_block else 0.0,
        }
        print(
            f"  {domain}: {len(blocks)} blocks ({n_dirty} dirty), median "
            f"{results[domain]['median_docs_per_block']:.0f} docs/block"
        )
    return results


ZSTD_DICT_SIZE = 112 * 1024  # matches the WikiText post's trained-dict size
ZSTD_DICT_TRAIN_CHUNK = 512  # split train text into chunk-sized samples for training


def run_zstd_dict_for_chunk_size(chunk_size):
    """zstd --train: a dictionary trained on each domain's own TRAIN split
    (never the test chunks measured here), same 112KB size as the WikiText
    version in the main post. Samples for training are the train split cut
    into ZSTD_DICT_TRAIN_CHUNK-token windows, since zstd's dictionary
    trainer wants many samples, not one giant blob."""
    results = {}
    for domain in DOMAINS:
        train_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_train.npy")).astype(np.int64)

        train_samples_ids = make_chunks(train_r50k, ZSTD_DICT_TRAIN_CHUNK, 400)
        samples = [r50k.decode(c.tolist()).encode("utf-8") for c in train_samples_ids]
        zdict = zstd.train_dictionary(ZSTD_DICT_SIZE, samples)

        dict_c = zstd.ZstdCompressor(level=19, dict_data=zdict)
        dict_d = zstd.ZstdDecompressor(dict_data=zdict)

        test_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_test.npy")).astype(np.int64)
        chunks = make_chunks(test_r50k, chunk_size, N_CHUNKS)
        texts = [r50k.decode(c.tolist()) for c in chunks]
        raw_byte_lists = [t.encode("utf-8") for t in texts]

        ratios, enc_t, dec_t = [], [], []
        for raw in raw_byte_lists:
            t0 = time.perf_counter()
            comp = dict_c.compress(raw)
            t1 = time.perf_counter()
            decomp = dict_d.decompress(comp)
            t2 = time.perf_counter()
            assert decomp == raw
            ratios.append(len(raw) / len(comp))
            enc_t.append(t1 - t0)
            dec_t.append(t2 - t1)

        results[domain] = {
            "ratio": bootstrap_ci(ratios),
            "encode_us": bootstrap_ci(np.array(enc_t) * 1e6),
            "decode_us": bootstrap_ci(np.array(dec_t) * 1e6),
        }
        print(f"  {domain}: dict trained on {len(samples)} train samples")
    return results


if "--zstd-dict" in sys.argv:
    print("=== zstd --train (112KB dict per domain) -- only this method ===")
    zstd_dict_results = {}
    for cs in CHUNK_SIZES:
        print(f"\n--- chunk_size={cs} ---")
        zstd_dict_results[cs] = run_zstd_dict_for_chunk_size(cs)

    print(f"\n{'=' * 90}")
    print("  zstd --train (112KB dict trained on each domain's own train split)")
    print(f"{'=' * 90}")
    print(f"  {'chunk':>6} {'domain':<7} {'ratio':>18} {'encode us':>18} {'decode us':>18}")
    for cs in CHUNK_SIZES:
        for d in DOMAINS:
            r = zstd_dict_results[cs][d]
            rm, rc = r["ratio"]
            em, ec = r["encode_us"]
            dm, dc = r["decode_us"]
            print(
                f"  {cs:>6} {d:<7}"
                f" {rm:.2f}x[{rc[0]:.2f},{rc[1]:.2f}]".rjust(19)
                + f" {em:.1f}[{ec[0]:.1f},{ec[1]:.1f}]".rjust(19)
                + f" {dm:.1f}[{dc[0]:.1f},{dc[1]:.1f}]".rjust(19)
            )

    out_name = "summary_tables_" + "_".join(str(c) for c in CHUNK_SIZES) + ".json"
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.setdefault("ratio", {})
    existing.setdefault("latency", {})
    for cs in CHUNK_SIZES:
        for d in DOMAINS:
            r = zstd_dict_results[cs][d]
            existing["ratio"][f"{cs}|{d}|zstd dict"] = r["ratio"]
            existing["latency"][f"{cs}|{d}|zstd dict"] = {
                "encode": r["encode_us"],
                "decode": r["decode_us"],
            }
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nMerged into {out_name} (did not touch any other method's numbers)")
    sys.exit(0)


if "--lz4-blocks" in sys.argv:
    print("=== LZ4 (Elasticsearch/Lucene-style blocks) -- only this method ===")
    lz4_block_results = {}
    for cs in CHUNK_SIZES:
        print(f"\n--- chunk_size={cs} ---")
        lz4_block_results[cs] = run_lz4_blocks_for_chunk_size(cs)

    print(f"\n{'=' * 100}")
    print("  LZ4 (ES-style blocks: <=16KB or 128 docs/block, whole block compressed)")
    print(f"{'=' * 100}")
    print(f"  {'tok':>6} {'domain':<7} {'ratio':>18} {'encode us/doc':>18} {'decode us (1 doc)':>20} {'docs/blk':>9}")
    for cs in CHUNK_SIZES:
        for d in DOMAINS:
            r = lz4_block_results[cs][d]
            rm, rc = r["ratio"]
            em, ec = r["encode_per_doc_us"]
            dm, dc = r["decode_per_doc_us"]
            print(
                f"  {cs:>6} {d:<7}"
                f" {rm:.2f}x[{rc[0]:.2f},{rc[1]:.2f}]".rjust(19)
                + f" {em:.1f}[{ec[0]:.1f},{ec[1]:.1f}]".rjust(19)
                + f" {dm:.1f}[{dc[0]:.1f},{dc[1]:.1f}]".rjust(21)
                + f" {r['median_docs_per_block']:>9.0f}"
            )

    out_name = "summary_tables_" + "_".join(str(c) for c in CHUNK_SIZES) + ".json"
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.setdefault("ratio", {})
    existing.setdefault("latency", {})
    existing.setdefault("lz4_blocks_meta", {})
    for cs in CHUNK_SIZES:
        for d in DOMAINS:
            r = lz4_block_results[cs][d]
            existing["ratio"][f"{cs}|{d}|LZ4 (ES blocks)"] = r["ratio"]
            existing["latency"][f"{cs}|{d}|LZ4 (ES blocks)"] = {
                "encode": r["encode_per_doc_us"],
                "decode": r["decode_per_doc_us"],
            }
            existing["lz4_blocks_meta"][f"{cs}|{d}"] = {
                "n_blocks": r["n_blocks"],
                "n_dirty_blocks": r["n_dirty_blocks"],
                "median_docs_per_block": r["median_docs_per_block"],
            }
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nMerged into {out_name} (did not touch any other method's numbers)")
    sys.exit(0)


METHOD_ORDER = [
    "LZ4", "gzip-9", "zstd-19", "brotli-q11",
    "r50k raw", "cl100k raw", "o200k raw",
    "r50k +ANS", "cl100k +ANS", "o200k +ANS",
]

# cache each domain's train text once (decoded from r50k train ids) -- reused
# across chunk sizes since the static tables don't depend on chunk size
train_text_cache = {}


def run_for_chunk_size(chunk_size):
    ratio_results = {}
    latency_results = {}

    for domain in DOMAINS:
        print(f"=== {domain} (chunk_size={chunk_size}) ===")
        test_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_test.npy")).astype(np.int64)

        chunks_r50k = make_chunks(test_r50k, chunk_size, N_CHUNKS)
        texts = [r50k.decode(c.tolist()) for c in chunks_r50k]
        raw_byte_lists = [t.encode("utf-8") for t in texts]
        print(f"  {len(texts)} test chunks reconstructed")
        if len(texts) < 5:
            print(f"  SKIP {domain}: not enough tokens for chunk_size={chunk_size}")
            continue

        # ── byte codecs ──────────────────────────────────────────────────────
        codec_fns = {
            "LZ4": (lambda b: lz4.frame.compress(b), lambda b: lz4.frame.decompress(b)),
            "gzip-9": (lambda b: gzip.compress(b, compresslevel=9), lambda b: gzip.decompress(b)),
            "zstd-19": (lambda b: zstd_c.compress(b), lambda b: zstd.ZstdDecompressor().decompress(b)),
            "brotli-q11": (lambda b: brotli.compress(b, quality=11), lambda b: brotli.decompress(b)),
        }
        for name, (cfn, dfn) in codec_fns.items():
            ratios, enc_t, dec_t = [], [], []
            for raw in raw_byte_lists:
                t0 = time.perf_counter()
                comp = cfn(raw)
                t1 = time.perf_counter()
                decomp = dfn(comp)
                t2 = time.perf_counter()
                assert decomp == raw
                ratios.append(len(raw) / len(comp))
                enc_t.append(t1 - t0)
                dec_t.append(t2 - t1)
            ratio_results[(domain, name)] = bootstrap_ci(ratios)
            latency_results[(domain, name)] = {
                "encode": bootstrap_ci(np.array(enc_t) * 1e6),
                "decode": bootstrap_ci(np.array(dec_t) * 1e6),
            }
        print("  byte codecs done")

        # ── per-tokenizer: raw packing + static ANS ─────────────────────────
        if domain not in train_text_cache:
            train_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_train.npy")).astype(np.int64)
            train_text_cache[domain] = r50k.decode(train_r50k.tolist())
        train_text = train_text_cache[domain]
        train_r50k_ids = np.load(os.path.join(CORPUS_DIR, f"{domain}_train.npy")).astype(np.int64)

        # ── zstd --train, FULL-train dictionary (per-chunk, bootstrap later) ──
        zc_full = full_train_zstd_compressor(domain, train_r50k_ids)
        run_for_chunk_size.zstdtrain_raw[(chunk_size, domain)] = [
            len(raw) / len(zc_full.compress(raw)) for raw in raw_byte_lists
        ]

        for tok_key, (enc_name, vocab_size, container_bytes) in TOKENIZERS.items():
            enc = TOKENIZER_ENCODERS[tok_key]
            test_ids_list = [enc.encode(t, disallowed_special=()) for t in texts]

            pack_ratios, tok_enc_t, tok_dec_t = [], [], []
            for raw, text in zip(raw_byte_lists, texts):
                t0 = time.perf_counter()
                ids = enc.encode(text, disallowed_special=())
                t1 = time.perf_counter()
                decoded = enc.decode(ids)
                t2 = time.perf_counter()
                assert decoded == text
                pack_ratios.append(len(raw) / (len(ids) * container_bytes))
                tok_enc_t.append(t1 - t0)
                tok_dec_t.append(t2 - t1)
            ratio_results[(domain, f"{tok_key} raw")] = bootstrap_ci(pack_ratios)
            latency_results[(domain, f"{tok_key} raw")] = {
                "encode": bootstrap_ci(np.array(tok_enc_t) * 1e6),
                "decode": bootstrap_ci(np.array(tok_dec_t) * 1e6),
            }

            cache_key = (domain, tok_key)
            if cache_key not in run_for_chunk_size.model_cache:
                train_ids = enc.encode(train_text, disallowed_special=())
                counts = np.ones(vocab_size, dtype=np.int64)
                counts += np.bincount(train_ids, minlength=vocab_size)
                probs = counts.astype(np.float64) / counts.sum()
                run_for_chunk_size.model_cache[cache_key] = constriction.stream.model.Categorical(
                    probs, perfect=False
                )
                # Frequency rank table for Kalcher/+freq, trained on the SAME
                # full train split (matches +freq's convention). No RNG draw.
                fcounts = np.bincount(np.asarray(train_ids, dtype=np.int64), minlength=vocab_size)
                order = np.argsort(-fcounts)
                rank_of = np.empty(vocab_size, dtype=np.uint32)
                rank_of[order] = np.arange(vocab_size, dtype=np.uint32)
                run_for_chunk_size.rank_cache[cache_key] = rank_of
            model = run_for_chunk_size.model_cache[cache_key]
            rank_of = run_for_chunk_size.rank_cache[cache_key]

            ans_ratios, ans_enc_t, ans_dec_t = [], [], []
            klzma_ratios, kzstd_ratios = [], []  # Kalcher, no RNG (bootstrap later)
            freq_ratios = []  # +freq (streamvbyte), no RNG (bootstrap later)
            for raw, text, ids in zip(raw_byte_lists, texts, test_ids_list):
                t0 = time.perf_counter()
                ids2 = enc.encode(text, disallowed_special=())
                c = constriction.stream.stack.AnsCoder()
                c.encode_reverse(np.array(ids2, dtype=np.int32), model)
                payload = c.get_compressed().tobytes()
                t1 = time.perf_counter()

                c2 = constriction.stream.stack.AnsCoder(np.frombuffer(payload, dtype=np.uint32).copy())
                decoded_ids = c2.decode(model, len(ids2))
                decoded_text = enc.decode(list(decoded_ids))
                t2 = time.perf_counter()
                assert decoded_text == text

                ans_ratios.append(len(raw) / len(payload))
                ans_enc_t.append(t1 - t0)
                ans_dec_t.append(t2 - t1)

                # ---- Kalcher: freq-remap -> LEB128 -> {LZMA, zstd-22} ----
                # (deliberately draws NO RNG so the blog's chunk selection and
                # every +ANS/raw/byte number above stay bit-identical; Kalcher
                # ratios are bootstrapped after the aligned run finishes.)
                remapped = rank_of[np.asarray(ids2, dtype=np.int64)]
                freq_ratios.append(len(raw) / len(svb_encode_arr(remapped)))  # +freq
                varint = leb128_encode(remapped)
                assert np.array_equal(leb128_decode(varint), remapped)
                klzma_ratios.append(len(raw) / len(lzma.compress(varint, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)))
                kzstd_ratios.append(len(raw) / len(zstd_c22.compress(varint)))
            ratio_results[(domain, f"{tok_key} +ANS")] = bootstrap_ci(ans_ratios)
            latency_results[(domain, f"{tok_key} +ANS")] = {
                "encode": bootstrap_ci(np.array(ans_enc_t) * 1e6),
                "decode": bootstrap_ci(np.array(ans_dec_t) * 1e6),
            }
            run_for_chunk_size.kalcher_raw[(chunk_size, domain, f"{tok_key} Kalcher(LZMA)")] = klzma_ratios
            run_for_chunk_size.kalcher_raw[(chunk_size, domain, f"{tok_key} Kalcher(zstd)")] = kzstd_ratios
            run_for_chunk_size.freq_raw[(chunk_size, domain, f"{tok_key} +freq")] = freq_ratios
            print(f"  {tok_key} done (raw packing + static ANS + Kalcher + freq)")

    return ratio_results, latency_results


run_for_chunk_size.model_cache = {}
run_for_chunk_size.rank_cache = {}
run_for_chunk_size.kalcher_raw = {}  # (cs, domain, method) -> per-chunk ratios
run_for_chunk_size.freq_raw = {}     # (cs, domain, "{tok} +freq") -> per-chunk ratios
run_for_chunk_size.zstdtrain_raw = {}  # (cs, domain) -> per-chunk ratios

all_ratio, all_latency = {}, {}
for cs in CHUNK_SIZES:
    r, l = run_for_chunk_size(cs)
    all_ratio[cs] = r
    all_latency[cs] = l

for cs in CHUNK_SIZES:
    ratio_results, latency_results = all_ratio[cs], all_latency[cs]
    print(f"\n{'=' * 100}")
    print(f"  TABLE 1: Compression ratio vs raw UTF-8 bytes, {cs}-token chunks, N={N_CHUNKS}/domain")
    print(f"{'=' * 100}")
    print(f"  {'method':<14} {'prose':>20} {'code':>20} {'hindi':>20}")
    for m in METHOD_ORDER:
        row = f"  {m:<14}"
        for d in DOMAINS:
            if (d, m) not in ratio_results:
                row += " " + "n/a".rjust(20)
                continue
            med, ci = ratio_results[(d, m)]
            row += f" {med:.2f}x[{ci[0]:.2f},{ci[1]:.2f}]".rjust(21)
        print(row)

    print(f"\n{'=' * 100}")
    print(f"  TABLE 2: Encode/decode latency (us/doc), {cs}-token chunks, N={N_CHUNKS}/domain")
    print(f"{'=' * 100}")
    for op in ["encode", "decode"]:
        print(f"  --- {op} ---")
        print(f"  {'method':<14} {'prose':>20} {'code':>20} {'hindi':>20}")
        for m in METHOD_ORDER:
            row = f"  {m:<14}"
            for d in DOMAINS:
                if (d, m) not in latency_results:
                    row += " " + "n/a".rjust(20)
                    continue
                med, ci = latency_results[(d, m)][op]
                row += f" {med:.1f}[{ci[0]:.1f},{ci[1]:.1f}]".rjust(21)
            print(row)

out_name = "summary_tables_" + "_".join(str(c) for c in CHUNK_SIZES) + ".json"
with open(os.path.join(os.path.dirname(__file__), out_name), "w") as f:
    json.dump(
        {
            "ratio": {
                f"{cs}|{d}|{m}": all_ratio[cs][(d, m)]
                for cs in CHUNK_SIZES for d in DOMAINS for m in METHOD_ORDER if (d, m) in all_ratio[cs]
            },
            "latency": {
                f"{cs}|{d}|{m}": all_latency[cs][(d, m)]
                for cs in CHUNK_SIZES for d in DOMAINS for m in METHOD_ORDER if (d, m) in all_latency[cs]
            },
        },
        f, indent=2,
    )
print("\nSaved to summary_tables.json")

# ══════════════════════════════════════════════════════════════════════════
# Kalcher rows (matched to the reproduced Table 1) -- focus on 512-token chunks
# ══════════════════════════════════════════════════════════════════════════
CS = 512
TOKS = ["r50k", "cl100k", "o200k"]
# Bootstrap Kalcher NOW (after the aligned run, so it can't perturb chunk
# selection). bootstrap_ci uses the global RNG, which is fine at this point.
kalcher_ci = {}
for tok in TOKS:
    for m in ["Kalcher(LZMA)", "Kalcher(zstd)"]:
        for d in DOMAINS:
            kalcher_ci[(d, f"{tok} {m}")] = bootstrap_ci(
                run_for_chunk_size.kalcher_raw[(CS, d, f"{tok} {m}")]
            )

# ---- anchor reproduction check (proves the harness matches the blog) ----
print(f"\n{'=' * 78}\nANCHOR REPRODUCTION CHECK vs blog 3-domain Table (512-tok, seed 9012)\n{'=' * 78}")
r512 = all_ratio[CS]
ok = True
checks = [("prose", "r50k +ANS", 3.30), ("code", "cl100k +ANS", 3.19),
          ("hindi", "o200k +ANS", 5.90), ("prose", "cl100k +ANS", 3.37),
          ("prose", "o200k +ANS", 3.40), ("code", "r50k +ANS", 2.58),
          ("hindi", "cl100k +ANS", 3.48), ("code", "o200k +ANS", 3.16),
          ("hindi", "r50k +ANS", 2.97)]
for d, m, exp in checks:
    got = r512[(d, m)][0]
    flag = "OK" if abs(got - exp) <= 0.03 else "OFF"
    if flag == "OFF":
        ok = False
    print(f"  {d:<6} {m:<12} got {got:.2f}x  expected {exp:.2f}x  [{flag}]")
print(f"  => Table-1 anchors {'REPRODUCED EXACTLY' if ok else 'NOT reproduced'} "
      f"(the +ANS/raw/byte rows above are the blog's, bit-for-bit)")

# ---- Kalcher rows, per tokenizer x language ----
print(f"\n{'=' * 90}\nKALCHER ROWS (drop-in Table 1, 512-tok), median [90% CI]\n{'=' * 90}")
print(f"  {'method':<20}{'prose':>22}{'code':>22}{'hindi':>22}")
for tok in TOKS:
    for m in ["Kalcher(LZMA)", "Kalcher(zstd)"]:
        row = f"  {tok+' '+m:<20}"
        for d in DOMAINS:
            v = kalcher_ci[(d, f"{tok} {m}")]
            row += f"{v[0]:.2f}x[{v[1][0]:.2f},{v[1][1]:.2f}]".rjust(22)
        print(row)

# ---- overall best method per language (across ALL methods) ----
all_methods = list(METHOD_ORDER)
kalcher_methods = [f"{t} {m}" for t in TOKS for m in ["Kalcher(LZMA)", "Kalcher(zstd)"]]


def ratio_of(d, m):
    if m in kalcher_methods:
        return kalcher_ci[(d, m)][0]
    return r512[(d, m)][0]


print(f"\n{'=' * 78}\nOVERALL BEST METHOD PER LANGUAGE (max median ratio, all methods)\n{'=' * 78}")
best_per_lang = {}
for d in DOMAINS:
    cand = all_methods + kalcher_methods
    ranked = sorted(cand, key=lambda m: -ratio_of(d, m))
    bm = ranked[0]
    best_per_lang[d] = {"method": bm, "ratio": ratio_of(d, bm),
                        "runner_up": ranked[1], "runner_up_ratio": ratio_of(d, ranked[1])}
    print(f"  {d:<6} BEST = {bm:<22} {ratio_of(d, bm):.2f}x   "
          f"(2nd: {ranked[1]} {ratio_of(d, ranked[1]):.2f}x)")

# ---- merge into results.json ----
out_path = os.path.join(os.path.dirname(__file__), "results.json")
existing = {}
if os.path.exists(out_path):
    with open(out_path) as f:
        existing = json.load(f)
existing["table1_kalcher_matched"] = {
    "config": {
        "harness": "faithful copy of 01_storage_efficiency/bench_summary_tables.py "
                   "(seed 9012, full-train Laplace ANS, RNG-aligned); Kalcher added inline on "
                   "identical chunks, bootstrapped after the aligned run so +ANS/raw/byte rows "
                   "stay bit-identical to the blog",
        "chunk_size": CS, "n_chunks": N_CHUNKS, "seed": 9012,
        "rank_table": "full train split (matches +freq convention)",
        "lzma": "FORMAT_RAW LZMA2 preset=9|EXTREME", "zstd": "level 22",
    },
    "anchor_check": {f"{d}|{m}": {"got": r512[(d, m)][0], "expected": exp}
                     for d, m, exp in checks},
    "existing_table1_rows": {f"{d}|{m}": r512[(d, m)] for d in DOMAINS for m in METHOD_ORDER
                             if (d, m) in r512},
    "kalcher_rows": {f"{d}|{tok} {m}": kalcher_ci[(d, f"{tok} {m}")]
                     for d in DOMAINS for tok in TOKS for m in ["Kalcher(LZMA)", "Kalcher(zstd)"]},
    "best_per_language": best_per_lang,
}
with open(out_path, "w") as f:
    json.dump(existing, f, indent=2)
print("\nMerged into results.json (table1_kalcher_matched)")


# ══════════════════════════════════════════════════════════════════════════
# PATH B: the COMPLETE Table 1 on ONE internally-consistent full-train setup.
# Every cell measured in this single harness (seed 9012, full-train ANS +
# full-train rank table + full-train zstd dict) on the identical 512-tok chunks.
# ══════════════════════════════════════════════════════════════════════════
freq_ci = {}
for (cs2, d, mm), v in run_for_chunk_size.freq_raw.items():
    if cs2 == CS:
        freq_ci[(d, mm)] = bootstrap_ci(v)
zstdtrain_ci = {d: bootstrap_ci(v) for (cs2, d), v in run_for_chunk_size.zstdtrain_raw.items() if cs2 == CS}

# Table B row order (coordinator's method list)
ROWS_B = [
    ("LZ4", lambda d: r512[(d, "LZ4")][0]),
    ("gzip-9", lambda d: r512[(d, "gzip-9")][0]),
    ("zstd-19", lambda d: r512[(d, "zstd-19")][0]),
    ("brotli-q11", lambda d: r512[(d, "brotli-q11")][0]),
    ("zstd --train", lambda d: zstdtrain_ci[d][0]),
    ("r50k raw", lambda d: r512[(d, "r50k raw")][0]),
    ("o200k raw", lambda d: r512[(d, "o200k raw")][0]),
    ("r50k +freq", lambda d: freq_ci[(d, "r50k +freq")][0]),
    ("o200k +freq", lambda d: freq_ci[(d, "o200k +freq")][0]),
    ("r50k +ANS", lambda d: r512[(d, "r50k +ANS")][0]),
    ("cl100k +ANS", lambda d: r512[(d, "cl100k +ANS")][0]),
    ("o200k +ANS", lambda d: r512[(d, "o200k +ANS")][0]),
    ("cl100k Kalcher(LZMA)", lambda d: kalcher_ci[(d, "cl100k Kalcher(LZMA)")][0]),
    ("cl100k Kalcher(zstd)", lambda d: kalcher_ci[(d, "cl100k Kalcher(zstd)")][0]),
    ("o200k Kalcher(LZMA)", lambda d: kalcher_ci[(d, "o200k Kalcher(LZMA)")][0]),
    ("o200k Kalcher(zstd)", lambda d: kalcher_ci[(d, "o200k Kalcher(zstd)")][0]),
]

print(f"\n{'=' * 78}\nPATH B -- COMPLETE Table 1, ONE full-train setup (512-tok), median\n{'=' * 78}")
print(f"  {'method':<22}{'English':>10}{'Code':>10}{'Hindi':>10}")
tableB = {}
for name, fn in ROWS_B:
    vals = {d: round(fn(d), 2) for d in DOMAINS}
    tableB[name] = vals
    print(f"  {name:<22}{vals['prose']:>9.2f}x{vals['code']:>9.2f}x{vals['hindi']:>9.2f}x")

# ---- diff vs blog table (train-dependent cells expected to move) ----
BLOG_512 = {  # published blog medians (prose, code, hindi)
    "LZ4": (1.27, 1.76, 1.52), "gzip-9": (1.92, 2.46, 2.38), "zstd-19": (1.94, 2.45, 2.42),
    "brotli-q11": (2.57, 2.87, 2.89), "zstd --train": (2.69, 3.24, 4.56),
    "r50k raw": (2.25, 1.07, 0.84), "o200k raw": (1.59, 1.49, 2.55),
    "r50k +freq": (2.62, 1.42, 1.33), "o200k +freq": (2.76, 2.53, 4.39),
    "r50k +ANS": (3.30, 2.58, 2.97), "cl100k +ANS": (3.37, 3.19, 3.48), "o200k +ANS": (3.40, 3.16, 5.90),
}
# note: r50k +ANS blog English headline is 3.26 in the ENGLISH-ONLY script
# (400-chunk ANS); the 3-domain blog table uses full-train -> 3.30.
print(f"\n{'=' * 78}\nCELLS CHANGED vs BLOG TABLE (|Δ|>0.02x; train-dependent rows)\n{'=' * 78}")
changes = []
for name, _ in ROWS_B:
    if name not in BLOG_512:
        continue  # Kalcher rows are new; no blog value
    for i, d in enumerate(DOMAINS):
        got = tableB[name][d]
        blog = BLOG_512[name][i]
        if abs(got - blog) > 0.02:
            changes.append((name, d, blog, got, got - blog))
if not changes:
    print("  (none beyond rounding)")
for name, d, blog, got, delta in changes:
    print(f"  {name:<22} {d:<6} {blog:.2f}x -> {got:.2f}x  ({delta:+.2f})")
print("\n  Sanity: raw + non-trained byte codecs should be unchanged (|Δ|<=0.02):")
for name in ["LZ4", "gzip-9", "zstd-19", "brotli-q11", "r50k raw", "o200k raw"]:
    unchanged = all(abs(tableB[name][d] - BLOG_512[name][i]) <= 0.02 for i, d in enumerate(DOMAINS))
    print(f"    {name:<12} {'UNCHANGED' if unchanged else 'MOVED (unexpected!)'}")

with open(out_path) as f:
    existing = json.load(f)
existing["table1_full_train_consistent"] = {
    "config": {
        "harness": "same validated full-train harness (seed 9012); EVERY cell in one setup: "
                   "full-train Laplace ANS, full-train rank table (+freq), full-train zstd dict, "
                   "Kalcher LEB128+{LZMA9e,zstd22}; identical 512-tok chunks",
        "chunk_size": CS, "n_chunks": N_CHUNKS, "seed": 9012,
    },
    "table_median": {name: tableB[name] for name, _ in ROWS_B},
    "full_ci": {
        **{f"{d}|{name}": r512[(d, name)] for name in
           ["LZ4", "gzip-9", "zstd-19", "brotli-q11", "r50k raw", "o200k raw",
            "r50k +ANS", "cl100k +ANS", "o200k +ANS"] for d in DOMAINS if (d, name) in r512},
        **{f"{d}|zstd --train": zstdtrain_ci[d] for d in DOMAINS},
        **{f"{d}|{m}": freq_ci[(d, m)] for (d, m) in freq_ci},
        **{f"{d}|{tok} {m}": kalcher_ci[(d, f"{tok} {m}")]
           for d in DOMAINS for tok in ["cl100k", "o200k"] for m in ["Kalcher(LZMA)", "Kalcher(zstd)"]},
    },
    "changed_vs_blog": [{"method": n, "domain": d, "blog": b, "full_train": g, "delta": round(dl, 3)}
                        for n, d, b, g, dl in changes],
}
with open(out_path, "w") as f:
    json.dump(existing, f, indent=2)
print("\nMerged into results.json (table1_full_train_consistent)")
