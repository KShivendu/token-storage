"""
Fixes a real methodology bug in the zstd --train compress benchmark: the
earlier script (bench_zstd_dict.py, now gone) apparently created a fresh
ZstdCompressor per call, forcing the ~112KB trained dictionary to be
re-digested every single time (confirmed separately: 22.6ms/call fresh vs
28us/call with a reused + precomputed compressor). This measures compress
correctly: one dictionary, precomputed once, one compressor object reused
across all 40 test chunks x 30 reps each (same robust methodology as the
other fixed benchmarks).
"""
import os
import sys
import numpy as np
import tiktoken
import zstandard as zstd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tnbench as T
from tnbench import timed_reps, load_ids

DOMAINS = ["prose", "code", "hindi"]
CHUNK_SIZE = 512
N_CHUNKS = 40
REPS = 30
RNG = np.random.default_rng(3344)

r50k = tiktoken.get_encoding("r50k_base")


def bootstrap_ci(values):
    return T.bootstrap_ci(values, RNG)


def make_chunks(test_arr, chunk_size, n_chunks):
    return T.make_chunks(test_arr, chunk_size, n_chunks, RNG)


results = {}
for domain in DOMAINS:
    print(f"=== {domain} ===", flush=True)
    train_r50k = load_ids(f"{domain}_train")
    test_r50k = load_ids(f"{domain}_test")

    chunks_r50k = make_chunks(test_r50k, CHUNK_SIZE, N_CHUNKS)
    texts = [r50k.decode(c.tolist()) for c in chunks_r50k]
    raw_byte_lists = [t.encode("utf-8") for t in texts]

    train_chunks_bytes = [
        r50k.decode(train_r50k[i: i + CHUNK_SIZE].tolist()).encode("utf-8")
        for i in range(0, min(len(train_r50k), CHUNK_SIZE * 400), CHUNK_SIZE)
    ]
    zdict = zstd.ZstdCompressionDict(zstd.train_dictionary(112 * 1024, train_chunks_bytes).as_bytes())
    zdict.precompute_compress(level=19)  # the actual fix: digest the dict ONCE
    zc = zstd.ZstdCompressor(level=19, dict_data=zdict)  # ONE reused compressor
    zd = zstd.ZstdDecompressor(dict_data=zdict)

    comp_t, decomp_t = [], []
    for raw in raw_byte_lists:
        comp_t.append(timed_reps(lambda raw=raw: zc.compress(raw)))
        c = zc.compress(raw)
        decomp_t.append(timed_reps(lambda c=c: zd.decompress(c)))

    results[(domain, "compress")] = bootstrap_ci(comp_t)
    results[(domain, "decompress")] = bootstrap_ci(decomp_t)
    print(f"  compress/decompress done", flush=True)

print(f"\n{'=' * 90}")
print("  zstd --train, FIXED (precomputed dict, reused compressor), chunk_size=512, median us")
print(f"{'=' * 90}")
for domain in DOMAINS:
    c = results[(domain, "compress")]
    d = results[(domain, "decompress")]
    print(f"  {domain:<8} compress={c[0]:>8.2f}us [{c[1][0]:.2f},{c[1][1]:.2f}]  decompress={d[0]:>7.2f}us [{d[1][0]:.2f},{d[1][1]:.2f}]")

import json
with open(os.path.join(os.path.dirname(__file__), "zstd_train_fixed_results.json"), "w") as f:
    json.dump({f"{d}|{m}": v for (d, m), v in results.items()}, f, indent=2)
print("\nSaved to zstd_train_fixed_results.json")
