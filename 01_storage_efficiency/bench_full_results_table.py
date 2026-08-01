"""
Rebuilds the "Full results" tables (storage efficiency min/median/max, latency
median/p99) on English (C4), 512-token chunks, matching the rest of the post's
methodology (40 chunks, seed 3344). Keeps per-chunk arrays so min/max/p99
across chunks are real, not derived/guessed.

Encode/decode latency per chunk is itself the median of 30 reps (same robust
methodology as the other fixed benchmarks this session), then min/median/p99
are taken across the 40 chunks.
"""
import os
import sys
import numpy as np
import tiktoken
import lz4.frame as lz4f
import gzip
import zstandard as zstd
import brotli
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import make_chunks as _make_chunks, timed_reps, pack3, unpack3, build_ans_model, load_ids

CHUNK_SIZE = 512
N_CHUNKS = 40
REPS = 30
RNG = np.random.default_rng(3344)

r50k = tiktoken.get_encoding("r50k_base")
TOKENIZERS = {"r50k": ("r50k_base", 50257), "cl100k": ("cl100k_base", 100277), "o200k": ("o200k_base", 200019)}

train_ids = load_ids("prose_train")
test_ids = load_ids("prose_test")
train_text = r50k.decode(train_ids[: 400 * CHUNK_SIZE].tolist())


def make_chunks(test_arr, chunk_size, n_chunks):
    return _make_chunks(test_arr, chunk_size, n_chunks, RNG)


chunks = make_chunks(test_ids, CHUNK_SIZE, N_CHUNKS)
texts = [r50k.decode(c.tolist()) for c in chunks]
raw_byte_lens = np.array([len(t.encode("utf-8")) for t in texts])
raw_byte_lists = [t.encode("utf-8") for t in texts]

results = {}  # name -> {"ratio": [...], "enc": [...], "dec": [...]}

# ── byte codecs ───────────────────────────────────────────────────────────
codecs = {
    "LZ4": (lambda b: lz4f.compress(b), lambda c: lz4f.decompress(c)),
    "gzip-9": (lambda b: gzip.compress(b, 9), lambda c: gzip.decompress(c)),
    "zstd-19": (lambda b: zstd.ZstdCompressor(level=19).compress(b), lambda c: zstd.ZstdDecompressor().decompress(c)),
    "brotli-q11": (lambda b: brotli.compress(b, quality=11), lambda c: brotli.decompress(c)),
}
zc19 = zstd.ZstdCompressor(level=19)
zd19 = zstd.ZstdDecompressor()
codecs["zstd-19"] = (lambda b: zc19.compress(b), lambda c: zd19.decompress(c))

for name, (comp, decomp) in codecs.items():
    ratios, encs, decs = [], [], []
    for raw in raw_byte_lists:
        encs.append(timed_reps(lambda raw=raw: comp(raw)))
        c = comp(raw)
        ratios.append(len(raw) / len(c))
        decs.append(timed_reps(lambda c=c: decomp(c)))
    results[name] = {"ratio": ratios, "enc": encs, "dec": decs}
    print(f"{name} done", flush=True)

# ── zstd --train (dict trained on this domain's train split, precomputed) ──
train_chunks_bytes = [
    r50k.decode(train_ids[i: i + CHUNK_SIZE].tolist()).encode("utf-8")
    for i in range(0, min(len(train_ids), CHUNK_SIZE * 400), CHUNK_SIZE)
]
zdict = zstd.ZstdCompressionDict(zstd.train_dictionary(112 * 1024, train_chunks_bytes).as_bytes())
zdict.precompute_compress(level=19)
zc_dict = zstd.ZstdCompressor(level=19, dict_data=zdict)
zd_dict = zstd.ZstdDecompressor(dict_data=zdict)
ratios, encs, decs = [], [], []
for raw in raw_byte_lists:
    encs.append(timed_reps(lambda raw=raw: zc_dict.compress(raw)))
    c = zc_dict.compress(raw)
    ratios.append(len(raw) / len(c))
    decs.append(timed_reps(lambda c=c: zd_dict.decompress(c)))
results["zstd --train"] = {"ratio": ratios, "enc": encs, "dec": decs}
print("zstd --train done", flush=True)

# ── per-tokenizer: raw packing + static ANS ─────────────────────────────────
for tok_key, (enc_name, vocab_size) in TOKENIZERS.items():
    enc = tiktoken.get_encoding(enc_name)
    fits_uint16 = vocab_size <= 65536
    train_tok_ids = enc.encode(train_text, disallowed_special=())
    model = build_ans_model(train_tok_ids, vocab_size)

    raw_ratios, raw_encs, raw_decs = [], [], []
    ans_ratios, ans_encs, ans_decs = [], [], []
    for raw, text in zip(raw_byte_lists, texts):
        ids = enc.encode(text, disallowed_special=())
        ids_arr = np.array(ids, dtype=np.int64)

        def raw_encode():
            ids2 = enc.encode(text, disallowed_special=())
            a = np.array(ids2, dtype=np.int64)
            return a.astype(np.uint16).tobytes() if fits_uint16 else pack3(a)

        raw_encs.append(timed_reps(raw_encode))
        packed = ids_arr.astype(np.uint16).tobytes() if fits_uint16 else pack3(ids_arr)
        raw_ratios.append(len(raw) / len(packed))
        if fits_uint16:
            raw_decs.append(timed_reps(lambda p=packed: np.frombuffer(p, dtype=np.uint16)))
        else:
            raw_decs.append(timed_reps(lambda p=packed, n=len(ids_arr): unpack3(p, n)))

        ids32 = ids_arr.astype(np.int32)

        def ans_encode_once(ids32=ids32):
            c = constriction.stream.stack.AnsCoder()
            c.encode_reverse(ids32, model)
            return c.get_compressed().tobytes()

        ans_encs.append(timed_reps(ans_encode_once))
        payload = ans_encode_once()
        ans_ratios.append(len(raw) / len(payload))

        payload_arr = np.frombuffer(payload, dtype=np.uint32).copy()

        def ans_decode_once(payload_arr=payload_arr, n=len(ids)):
            c2 = constriction.stream.stack.AnsCoder(payload_arr.copy())
            return c2.decode(model, n)

        ans_decs.append(timed_reps(ans_decode_once))

    results[f"{tok_key} raw"] = {"ratio": raw_ratios, "enc": raw_encs, "dec": raw_decs}
    results[f"{tok_key} +ANS"] = {"ratio": ans_ratios, "enc": ans_encs, "dec": ans_decs}
    print(f"{tok_key} raw/+ANS done", flush=True)

print(f"\n{'='*100}")
print("STORAGE EFFICIENCY (min / median / max), English (C4), 512-token chunks")
print(f"{'='*100}")
print(f"{'Method':<16}{'min':>10}{'median':>10}{'max':>10}")
for name, r in results.items():
    ratios = np.array(r["ratio"])
    print(f"{name:<16}{ratios.min():>9.2f}x{np.median(ratios):>9.2f}x{ratios.max():>9.2f}x")

print(f"\n{'='*100}")
print("LATENCY (median / p99), English (C4), 512-token chunks")
print(f"{'='*100}")
print(f"{'Method':<16}{'enc median':>12}{'enc p99':>10}{'dec median':>12}{'dec p99':>10}")
for name, r in results.items():
    enc = np.array(r["enc"])
    dec = np.array(r["dec"])
    print(f"{name:<16}{np.median(enc):>10.1f}us{np.percentile(enc,99):>8.1f}us{np.median(dec):>10.1f}us{np.percentile(dec,99):>8.1f}us")

import json
with open(os.path.join(os.path.dirname(__file__), "full_results_table.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to full_results_table.json")
