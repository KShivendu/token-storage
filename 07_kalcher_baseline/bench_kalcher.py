"""
Kalcher 2026 baseline ("Frequency-Ordered Tokenization for Better Text
Compression", arXiv:2602.22958) head-to-head against this repo's token-native
methods, on the same corpora / 512-token chunks / train-test split / bootstrap-
CI / median-of-30-reps latency methodology as the rest of the suite.

Kalcher = freq-remap -> LEB128 -> {LZMA, zstd-22}. Contrast with this repo's
+freq = freq-remap -> streamvbyte (no general compressor): Kalcher buys extra
ratio with a general compressor but pays a much slower decode.

Methods (r50k token IDs): raw (uint16 packing), +freq (streamvbyte), +ANS
(static shared unigram ANS), Kalcher(LZMA), Kalcher(zstd-22). Per method:
compression ratio vs UTF-8 (median + bootstrap CI over 40 chunks), plus agent-
facing decode latency (stops at token IDs, no detokenize) and encode latency.
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
    load_ids, make_chunks, bootstrap_ci, timed_reps,
    leb128_encode, leb128_decode, svb_encode_arr, svb_decode_arr,
    build_rank_table, build_ans_model, zstd_c22, zstd_d, LZMA_FILTERS,
)

DOMAINS = ["prose", "code", "hindi"]
CHUNK_SIZE = 512
N_CHUNKS = 40
REPS = 30
VOCAB = 50257  # r50k
ANS_TRAIN_TOKENS = 400 * CHUNK_SIZE
SEED = 3344

r50k = tiktoken.get_encoding("r50k_base")


def main():
    rng = np.random.default_rng(SEED)
    results = {}  # (domain, method, metric) -> ci tuple / value
    for domain in DOMAINS:
        print(f"=== {domain} ===", flush=True)
        train = load_ids(f"{domain}_train")
        test = load_ids(f"{domain}_test")

        # +freq / Kalcher rank table: full train split, descending frequency
        rank_of, token_of_rank = build_rank_table(train, VOCAB)
        # static ANS model: first 400*512 train tokens (matches repo +ANS)
        ans_model = build_ans_model(train[:ANS_TRAIN_TOKENS], VOCAB)

        chunks = make_chunks(test, CHUNK_SIZE, N_CHUNKS, rng)
        texts = [r50k.decode(c.tolist()) for c in chunks]
        raw_lens = [len(t.encode("utf-8")) for t in texts]

        acc = {
            m: {"ratio": [], "enc": [], "dec": []}
            for m in ["raw", "+freq", "+ANS", "Kalcher(LZMA)", "Kalcher(zstd)"]
        }

        for chunk, raw_len in zip(chunks, raw_lens):
            ids = chunk.astype(np.int64)
            n = len(ids)
            remapped = rank_of[ids]  # frequency ranks (uint32)

            # ---- raw: r50k uint16 packing ----
            packed = ids.astype(np.uint16).tobytes()
            acc["raw"]["ratio"].append(raw_len / len(packed))
            acc["raw"]["enc"].append(timed_reps(lambda i=ids: i.astype(np.uint16).tobytes()))
            acc["raw"]["dec"].append(timed_reps(lambda p=packed: np.frombuffer(p, dtype=np.uint16)))

            # ---- +freq: freq-remap -> streamvbyte ----
            svb_payload = svb_encode_arr(remapped)
            assert np.array_equal(svb_decode_arr(svb_payload, n), remapped)
            acc["+freq"]["ratio"].append(raw_len / len(svb_payload))
            acc["+freq"]["enc"].append(timed_reps(lambda i=ids: svb_encode_arr(rank_of[i])))

            def freq_decode(p=svb_payload, n=n):
                return token_of_rank[svb_decode_arr(p, n)]

            acc["+freq"]["dec"].append(timed_reps(freq_decode))

            # ---- +ANS ----
            ids32 = ids.astype(np.int32)

            def ans_encode_once(ids32=ids32):
                c = constriction.stream.stack.AnsCoder()
                c.encode_reverse(ids32, ans_model)
                return c.get_compressed().tobytes()

            ans_payload = ans_encode_once()
            ans_arr = np.frombuffer(ans_payload, dtype=np.uint32).copy()

            def ans_decode_once(a=ans_arr, n=n):
                c2 = constriction.stream.stack.AnsCoder(a.copy())
                return c2.decode(ans_model, n)

            assert np.array_equal(np.asarray(ans_decode_once()), ids32)
            acc["+ANS"]["ratio"].append(raw_len / len(ans_payload))
            acc["+ANS"]["enc"].append(timed_reps(ans_encode_once))
            acc["+ANS"]["dec"].append(timed_reps(ans_decode_once))

            # ---- Kalcher: freq-remap -> LEB128 -> {LZMA, zstd-22} ----
            varint = leb128_encode(remapped)
            assert np.array_equal(leb128_decode(varint), remapped), "LEB128 round-trip"

            lzma_payload = lzma.compress(varint, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
            zstd_payload = zstd_c22.compress(varint)
            acc["Kalcher(LZMA)"]["ratio"].append(raw_len / len(lzma_payload))
            acc["Kalcher(zstd)"]["ratio"].append(raw_len / len(zstd_payload))

            def kalcher_lzma_encode(i=ids):
                v = leb128_encode(rank_of[i])
                return lzma.compress(v, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)

            def kalcher_zstd_encode(i=ids):
                return zstd_c22.compress(leb128_encode(rank_of[i]))

            acc["Kalcher(LZMA)"]["enc"].append(timed_reps(kalcher_lzma_encode))
            acc["Kalcher(zstd)"]["enc"].append(timed_reps(kalcher_zstd_encode))

            def kalcher_lzma_decode(p=lzma_payload):
                v = lzma.decompress(p, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
                return token_of_rank[leb128_decode(v)]

            def kalcher_zstd_decode(p=zstd_payload):
                return token_of_rank[leb128_decode(zstd_d.decompress(p))]

            assert np.array_equal(kalcher_lzma_decode(), ids)
            assert np.array_equal(kalcher_zstd_decode(), ids)
            acc["Kalcher(LZMA)"]["dec"].append(timed_reps(kalcher_lzma_decode))
            acc["Kalcher(zstd)"]["dec"].append(timed_reps(kalcher_zstd_decode))

        for m, d in acc.items():
            results[(domain, m, "ratio")] = bootstrap_ci(d["ratio"], rng)
            results[(domain, m, "encode")] = bootstrap_ci(d["enc"], rng)
            results[(domain, m, "decode")] = bootstrap_ci(d["dec"], rng)
        for m in acc:
            r = results[(domain, m, "ratio")]
            dc = results[(domain, m, "decode")]
            print(f"  {m:<14} ratio={r[0]:.2f}x  decode={dc[0]:.1f}us", flush=True)

    methods = ["raw", "+freq", "+ANS", "Kalcher(LZMA)", "Kalcher(zstd)"]
    for title, metric, unit in [
        ("COMPRESSION RATIO vs UTF-8 (median [90% bootstrap CI]), 512-token chunks", "ratio", "x"),
        ("DECODE latency to token IDs (median us [90% CI]), 512-token chunks, median-of-30-reps", "decode", ""),
        ("ENCODE latency from token IDs (median us [90% CI]), 512-token chunks", "encode", ""),
    ]:
        print(f"\n{'='*90}\n{title}\n{'='*90}")
        print(f"  {'method':<16}" + "".join(f"{d:>22}" for d in DOMAINS))
        for m in methods:
            row = f"  {m:<16}"
            for d in DOMAINS:
                v = results[(d, m, metric)]
                fmt = ".2f" if metric == "ratio" else ".1f"
                row += f"{v[0]:{fmt}}{unit}[{v[1][0]:{fmt}},{v[1][1]:{fmt}}]".rjust(22)
            print(row)

    out = {
        "config": {
            "tokenizer": "r50k", "chunk_size": CHUNK_SIZE, "n_chunks": N_CHUNKS,
            "reps": REPS, "seed": SEED, "ans_train_tokens": ANS_TRAIN_TOKENS,
            "lzma": "FORMAT_RAW LZMA2 preset=9|EXTREME", "zstd": "level 22",
            "note": "decode/encode are agent-facing (token IDs <-> storage), no (de)tokenize; ratio excludes any length header",
        },
        "results": {f"{d}|{m}|{k}": v for (d, m, k), v in results.items()},
    }
    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results.json")


if __name__ == "__main__":
    main()
