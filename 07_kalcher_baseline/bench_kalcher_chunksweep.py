"""
Chunk-size sweep for the token-native methods, extending the 512-token results
in bench_kalcher.py to LONGER chunks: 512 / 1024 / 2048 / 4096 tokens.

The blog's existing sweep (token-storage-extra.mdx, via
01_storage_efficiency/bench_summary_tables.py) only went 256 / 512 / 2000 and
did not include Kalcher. This adds Kalcher(LZMA) and Kalcher(zstd) alongside
r50k raw / +freq / +ANS across the larger sizes.

Table convention is kept IDENTICAL to bench_kalcher.py so the 512 column
reproduces the committed 512 numbers and everything is comparable:
  - r50k token IDs (no re-tokenization), seed 3344, 40 held-out test chunks.
  - Frequency rank table (shared by +freq and Kalcher): each domain's FULL
    train split.
  - Static ANS model: first 400*512 train tokens (a fixed shared table, held
    constant across chunk sizes -- the entropy model does not depend on chunk
    size, so this isolates the chunk-size effect to the codecs themselves).
  - Ratio vs UTF-8, median + bootstrap CI, no length header.

Hypothesis under test: longer chunks give LZ77 more repetition to exploit, so
Kalcher (varint -> general compressor) should gain ratio with size while +ANS /
+freq (memoryless order-0 models) stay ~flat -- widening Kalcher's edge,
especially on repetitive code.
"""
import os
import sys
import json
import lzma
import numpy as np
import tiktoken
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import (
    load_ids, make_chunks, bootstrap_ci,
    leb128_encode, svb_encode_arr, build_rank_table, build_ans_model,
    zstd_c22, LZMA_FILTERS,
)

DOMAINS = ["prose", "code", "hindi"]
CHUNK_SIZES = [512, 1024, 2048, 4096]
N_CHUNKS = 40
VOCAB = 50257
ANS_TRAIN_TOKENS = 400 * 512  # fixed shared table, same as bench_kalcher.py
SEED = 3344
METHODS = ["raw", "+freq", "+ANS", "Kalcher(LZMA)", "Kalcher(zstd)"]

r50k = tiktoken.get_encoding("r50k_base")


def main():
    rng = np.random.default_rng(SEED)
    results = {}  # (domain, chunk_size, method) -> ratio ci
    for domain in DOMAINS:
        print(f"=== {domain} ===", flush=True)
        train = load_ids(f"{domain}_train")
        test = load_ids(f"{domain}_test")

        rank_of, _ = build_rank_table(train, VOCAB)
        ans_model = build_ans_model(train[:ANS_TRAIN_TOKENS], VOCAB)

        for cs in CHUNK_SIZES:
            chunks = make_chunks(test, cs, N_CHUNKS, rng)
            texts = [r50k.decode(c.tolist()) for c in chunks]
            raw_lens = [len(t.encode("utf-8")) for t in texts]

            acc = {m: [] for m in METHODS}
            for chunk, raw_len in zip(chunks, raw_lens):
                ids = chunk.astype(np.int64)
                remapped = rank_of[ids]

                # raw uint16
                acc["raw"].append(raw_len / (len(ids) * 2))

                # +freq: streamvbyte
                svb_payload = svb_encode_arr(remapped)
                acc["+freq"].append(raw_len / len(svb_payload))

                # +ANS
                c = constriction.stream.stack.AnsCoder()
                c.encode_reverse(ids.astype(np.int32), ans_model)
                acc["+ANS"].append(raw_len / len(c.get_compressed().tobytes()))

                # Kalcher: LEB128 -> {LZMA, zstd}
                varint = leb128_encode(remapped)
                lzma_payload = lzma.compress(varint, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
                zstd_payload = zstd_c22.compress(varint)
                acc["Kalcher(LZMA)"].append(raw_len / len(lzma_payload))
                acc["Kalcher(zstd)"].append(raw_len / len(zstd_payload))

            for m in METHODS:
                results[(domain, cs, m)] = bootstrap_ci(acc[m], rng)
            row = "  ".join(f"{m}={results[(domain,cs,m)][0]:.2f}x" for m in METHODS)
            print(f"  cs={cs:>5}: {row}", flush=True)

    # ── tables ──
    for domain in DOMAINS:
        print(f"\n{'='*90}\nRATIO vs UTF-8 by chunk size -- {domain} (r50k, median [90% CI])\n{'='*90}")
        print(f"  {'method':<16}" + "".join(f"{str(cs)+'tok':>17}" for cs in CHUNK_SIZES))
        for m in METHODS:
            row = f"  {m:<16}"
            for cs in CHUNK_SIZES:
                v = results[(domain, cs, m)]
                row += f"{v[0]:.2f}x[{v[1][0]:.2f},{v[1][1]:.2f}]".rjust(17)
            print(row)
        # Kalcher-vs-ANS gap trend
        gaps = [results[(domain, cs, "Kalcher(LZMA)")][0] - results[(domain, cs, "+ANS")][0]
                for cs in CHUNK_SIZES]
        print("  Kalcher(LZMA) - +ANS gap: " + "  ".join(
            f"{cs}:{g:+.2f}" for cs, g in zip(CHUNK_SIZES, gaps)))

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing["chunksize_sweep_ratio"] = {
        "config": {"tokenizer": "r50k", "seed": SEED, "n_chunks": N_CHUNKS,
                   "chunk_sizes": CHUNK_SIZES, "ans_train_tokens": ANS_TRAIN_TOKENS,
                   "rank_table": "full train split", "lzma": "FORMAT_RAW LZMA2 preset=9|EXTREME",
                   "zstd": "level 22", "note": "table convention identical to bench_kalcher.py 512 results"},
        "ratio": {f"{d}|{cs}|{m}": results[(d, cs, m)]
                  for d in DOMAINS for cs in CHUNK_SIZES for m in METHODS},
    }
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print("\nMerged into results.json (chunksize_sweep_ratio)")


if __name__ == "__main__":
    main()
