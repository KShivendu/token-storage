"""
v2 of bench_agent_mode.py: fixes a real measurement bug the user caught (single
perf_counter() shots on sub-microsecond/few-microsecond ops are dominated by
GC/scheduler noise, not the operation itself -- e.g. r50k raw unpack measured
slower than LZ4 decompress, which made no sense for a zero-copy reinterpret vs
real decompression). Fix: for every FAST op (decompress_only, raw_unpack_only,
ans_decode_only), time REPS=30 back-to-back calls per chunk and take the median
of those reps as that chunk's cost, then bootstrap across the 40 chunks as
before. tokenize_only stays single-shot (100s of us, not noise-sensitive at
this scale) for consistency with the rest of the post's methodology.
"""
import os
import sys
import time
import numpy as np
import tiktoken
import lz4.frame as lz4f
import gzip
import zstandard as zstd
import brotli
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tnbench as T
from tnbench import timed_reps, pack3, unpack3, build_ans_model, load_ids

DOMAINS = ["prose", "code", "hindi"]
CHUNK_SIZE = 512
N_CHUNKS = 40
REPS = 30
TOKENIZERS = {"r50k": ("r50k_base", 50257), "cl100k": ("cl100k_base", 100277), "o200k": ("o200k_base", 200019)}
RNG = np.random.default_rng(3344)

# thin RNG-bound wrappers over the shared harness (keeps call sites + RNG order)
def bootstrap_ci(values):
    return T.bootstrap_ci(values, RNG)


def make_chunks(test_arr, chunk_size, n_chunks):
    return T.make_chunks(test_arr, chunk_size, n_chunks, RNG)


r50k = tiktoken.get_encoding("r50k_base")
zstd_c19 = zstd.ZstdCompressor(level=19)
zstd_d = zstd.ZstdDecompressor()


results = {}
for domain in DOMAINS:
    print(f"=== {domain} ===", flush=True)
    train_r50k = load_ids(f"{domain}_train")
    test_r50k = load_ids(f"{domain}_test")
    train_text = r50k.decode(train_r50k.tolist())

    chunks_r50k = make_chunks(test_r50k, CHUNK_SIZE, N_CHUNKS)
    texts = [r50k.decode(c.tolist()) for c in chunks_r50k]
    raw_byte_lists = [t.encode("utf-8") for t in texts]

    # ---- byte codecs: compress (write side) + decompress (read side) ----
    codecs = {
        "LZ4": (lambda b: lz4f.compress(b), lambda c: lz4f.decompress(c)),
        "gzip-9": (lambda b: gzip.compress(b, 9), lambda c: gzip.decompress(c)),
        "zstd-19": (lambda b: zstd_c19.compress(b), lambda c: zstd_d.decompress(c)),
        "brotli-q11": (lambda b: brotli.compress(b, quality=11), lambda c: brotli.decompress(c)),
    }
    # Time compress (the write-side cost) separately, then compress ONCE per
    # chunk and repeatedly decompress that fixed payload (the read-side cost).
    for name, (comp, decomp) in codecs.items():
        comp_t, dec_t = [], []
        for raw in raw_byte_lists:
            comp_t.append(timed_reps(lambda r=raw: comp(r)))
            c = comp(raw)
            dec_t.append(timed_reps(lambda c=c: decomp(c)))
        results[(domain, name, "compress_only")] = bootstrap_ci(comp_t)
        results[(domain, name, "decompress_only")] = bootstrap_ci(dec_t)
        print(f"  {name} compress/decompress done", flush=True)

    # zstd --train: 112KB dict trained on this domain's train split
    train_chunks_bytes = [
        r50k.decode(train_r50k[i: i + CHUNK_SIZE].tolist()).encode("utf-8")
        for i in range(0, min(len(train_r50k), CHUNK_SIZE * 400), CHUNK_SIZE)
    ]
    zdict = zstd.ZstdCompressionDict(zstd.train_dictionary(112 * 1024, train_chunks_bytes).as_bytes())
    zc_dict = zstd.ZstdCompressor(level=19, dict_data=zdict)
    zd_dict = zstd.ZstdDecompressor(dict_data=zdict)
    comp_t, dec_t = [], []
    for raw in raw_byte_lists:
        comp_t.append(timed_reps(lambda r=raw: zc_dict.compress(r)))
        c = zc_dict.compress(raw)
        dec_t.append(timed_reps(lambda c=c: zd_dict.decompress(c)))
    results[(domain, "zstd --train", "compress_only")] = bootstrap_ci(comp_t)
    results[(domain, "zstd --train", "decompress_only")] = bootstrap_ci(dec_t)
    print("  zstd --train compress/decompress done", flush=True)

    # ---- per-tokenizer: tokenize-only (single-shot, large enough), raw
    # unpack-only (robust), ANS decode-only (robust) ----
    for tok_key, (enc_name, vocab_size) in TOKENIZERS.items():
        enc = tiktoken.get_encoding(enc_name)
        train_ids = enc.encode(train_text, disallowed_special=())
        model = build_ans_model(train_ids, vocab_size)

        tok_t, unpack_t, ans_dec_t = [], [], []
        for text in texts:
            ids = enc.encode(text, disallowed_special=())
            ids_arr = np.array(ids, dtype=np.int64)

            t0 = time.perf_counter()
            retok = enc.encode(text, disallowed_special=())
            t1 = time.perf_counter()
            tok_t.append((t1 - t0) * 1e6)

            if tok_key == "r50k":
                packed = ids_arr.astype(np.uint16).tobytes()
                unpack_t.append(timed_reps(lambda p=packed: np.frombuffer(p, dtype=np.uint16)))
            else:
                packed = pack3(ids_arr)
                n = len(ids_arr)
                unpack_t.append(timed_reps(lambda p=packed, n=n: unpack3(p, n)))

            c = constriction.stream.stack.AnsCoder()
            c.encode_reverse(np.array(ids, dtype=np.int32), model)
            payload = np.frombuffer(c.get_compressed().tobytes(), dtype=np.uint32).copy()
            n_ids = len(ids)

            def ans_decode_once(payload=payload, n_ids=n_ids):
                c2 = constriction.stream.stack.AnsCoder(payload.copy())
                return c2.decode(model, n_ids)

            # sanity check once per chunk (not timed)
            assert list(ans_decode_once()) == ids
            ans_dec_t.append(timed_reps(ans_decode_once))

        results[(domain, tok_key, "tokenize_only")] = bootstrap_ci(tok_t)
        results[(domain, tok_key, "raw_unpack_only")] = bootstrap_ci(unpack_t)
        results[(domain, tok_key, "ans_decode_only")] = bootstrap_ci(ans_dec_t)
        print(f"  {tok_key} tokenize/unpack/ans-decode done", flush=True)

print(f"\n{'=' * 100}")
print("  Agent-decode components (robust, median-of-30-reps per chunk), chunk_size=512, median us")
print(f"{'=' * 100}")
for domain in DOMAINS:
    print(f"\n-- {domain} --")
    for name in ["LZ4", "gzip-9", "zstd-19", "brotli-q11", "zstd --train"]:
        cr = results[(domain, name, "compress_only")]
        dr = results[(domain, name, "decompress_only")]
        print(f"  {name:<14} compress_only={cr[0]:>9.2f}us  decompress_only={dr[0]:>7.2f}us")
    for tok_key in TOKENIZERS:
        t = results[(domain, tok_key, "tokenize_only")]
        u = results[(domain, tok_key, "raw_unpack_only")]
        a = results[(domain, tok_key, "ans_decode_only")]
        print(f"  {tok_key:<8} tokenize_only={t[0]:>8.2f}us  raw_unpack_only={u[0]:>7.3f}us  ans_decode_only={a[0]:>7.3f}us")

import json
with open(os.path.join(os.path.dirname(__file__), "agent_mode_results_v2.json"), "w") as f:
    json.dump({f"{d}|{n}|{m}": v for (d, n, m), v in results.items()}, f, indent=2)
print("\nSaved to agent_mode_results_v2.json")
