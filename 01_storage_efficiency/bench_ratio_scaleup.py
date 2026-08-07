"""
Scale-up robustness check for the Table-1 compression RATIOS.

Reviewer concern: the headline median compression ratios (Table 1, 512-token
chunks) are measured on only N=40 test chunks per domain (~20k tokens), which
sounds too small. But data/corpus/{domain}_test.npy holds ~1.5M test tokens per
domain (~2900 non-overlapping 512-token chunks). This script re-runs the SAME
ratio pipeline (same helpers, same seed convention, same 512-token chunks) at
several chunk counts so that OLD (N=40) vs NEW (large N) differ ONLY in N.

We reuse tnbench's exact codec/remap/varint helpers (no reimplementation) and
the same full-train zstd-dict recipe used by the Table-1 producer
(07_kalcher_baseline/bench_kalcher_table1_matched.py). Ratio only -- no latency,
so we can afford the slow codecs (brotli-q11, LZMA, zstd --train) at large N.

Byte codecs are tokenizer-independent (measured on the r50k-decoded UTF-8 text).
Token-native methods are reported at o200k (per the reviewer ask); the pipeline
is identical for the other tokenizers if you extend TOKS.

Usage:
    uv run python 01_storage_efficiency/bench_ratio_scaleup.py \
        [--domains prose,code,hindi] [--ns 40,500,2000] [--full]
"""
import argparse
import gzip
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import (
    load_ids, make_chunks, build_ans_model, build_rank_table,
    leb128_encode, svb_encode_arr, zstd_c22, LZMA_FILTERS, pack3,
    full_train_zstd_dict,
)

CHUNK_SIZE = 512
SEED = 9012  # same seed convention as the Table-1 producer
# o200k is the reviewer-requested tokenizer for the token-native methods.
TOK = "o200k"
TOK_ENC_NAME = "o200k_base"
VOCAB = 200019
CONTAINER_BYTES = 3  # o200k ids need 3 bytes

zstd_c19 = zstd.ZstdCompressor(level=19)
r50k = tiktoken.get_encoding("r50k_base")   # corpus is stored in r50k ids
enc = tiktoken.get_encoding(TOK_ENC_NAME)   # target tokenizer for token methods


def byte_codec_ratios(texts, raw_lens, zc_dict):
    """Median ratio vs UTF-8 for each byte codec (tokenizer-independent)."""
    raws = [t.encode("utf-8") for t in texts]
    out = {}
    codecs = {
        "LZ4": lambda b: lz4.frame.compress(b),
        "gzip-9": lambda b: gzip.compress(b, compresslevel=9),
        "zstd-19": lambda b: zc19_compress(b),
        "brotli-q11": lambda b: brotli.compress(b, quality=11),
        "zstd --train": lambda b: zc_dict.compress(b),
    }
    for name, cfn in codecs.items():
        rs = [rl / len(cfn(raw)) for raw, rl in zip(raws, raw_lens)]
        out[name] = float(np.median(rs))
    return out


def zc19_compress(b):
    return zstd_c19.compress(b)


def token_ratios(texts, raw_lens, rank_of, model):
    """Median ratio vs UTF-8 for each token-native method at the target tokenizer."""
    ids_list = [np.array(enc.encode(t, disallowed_special=()), dtype=np.int64) for t in texts]

    raw_rs, lz4_rs, freq_rs, ans_rs, klzma_rs, kzstd_rs = [], [], [], [], [], []
    for ids, rl in zip(ids_list, raw_lens):
        # raw packing (3 bytes/id for o200k)
        packed = pack3(ids) if CONTAINER_BYTES == 3 else ids.astype(np.uint16).tobytes()
        raw_rs.append(rl / len(packed))
        # +lz4 over packed ids
        lz4_rs.append(rl / len(lz4.frame.compress(packed)))
        # +freq: freq-remap -> streamvbyte
        remapped = rank_of[ids]
        freq_rs.append(rl / len(svb_encode_arr(remapped)))
        # +ANS: static unigram ANS
        c = constriction.stream.stack.AnsCoder()
        c.encode_reverse(ids.astype(np.int32), model)
        ans_rs.append(rl / len(c.get_compressed().tobytes()))
        # Kalcher: freq-remap -> LEB128 -> {LZMA, zstd-22}
        varint = leb128_encode(remapped)
        klzma_rs.append(rl / len(lzma.compress(varint, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)))
        kzstd_rs.append(rl / len(zstd_c22.compress(varint)))

    return {
        f"{TOK} raw": float(np.median(raw_rs)),
        f"{TOK} +lz4": float(np.median(lz4_rs)),
        f"{TOK} +freq": float(np.median(freq_rs)),
        f"{TOK} +ANS": float(np.median(ans_rs)),
        f"{TOK} Kalcher(LZMA)": float(np.median(klzma_rs)),
        f"{TOK} Kalcher(zstd)": float(np.median(kzstd_rs)),
    }


def run_domain(domain, ns):
    """Return {N: {method: median_ratio}} for one domain, plus per-N wall time."""
    print(f"\n=== {domain} ===", flush=True)
    t_setup = time.perf_counter()
    train_r50k = load_ids(f"{domain}_train")
    train_text = r50k.decode(train_r50k.tolist())
    train_ids = enc.encode(train_text, disallowed_special=())
    rank_of, _ = build_rank_table(train_ids, VOCAB)
    model = build_ans_model(train_ids, VOCAB)
    # zstd --train dict: full train split, r50k-decoded 512-token windows (shared recipe)
    zc_dict = full_train_zstd_dict(train_r50k, r50k)[0]
    print(f"  setup (o200k train encode + rank/ANS + zstd-dict): {time.perf_counter()-t_setup:.1f}s", flush=True)

    test_r50k = load_ids(f"{domain}_test")
    results, times = {}, {}
    for n in ns:
        rng = np.random.default_rng(SEED)  # reset per N so each N samples independently
        chunks = make_chunks(test_r50k, CHUNK_SIZE, n, rng)
        texts = [r50k.decode(c.tolist()) for c in chunks]
        raw_lens = [len(t.encode("utf-8")) for t in texts]
        t0 = time.perf_counter()
        row = {}
        row.update(byte_codec_ratios(texts, raw_lens, zc_dict))
        row.update(token_ratios(texts, raw_lens, rank_of, model))
        dt = time.perf_counter() - t0
        results[n] = row
        times[n] = dt
        print(f"  N={len(chunks):<5} ({len(chunks)} chunks) ratios computed in {dt:.1f}s", flush=True)
    return results, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="prose")
    ap.add_argument("--ns", default="40,500,2000")
    ap.add_argument("--full", action="store_true", help="add the max available N (~all test chunks)")
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    ns = [int(x) for x in args.ns.split(",") if x.strip()]

    METHODS = [
        "LZ4", "gzip-9", "zstd-19", "brotli-q11", "zstd --train",
        f"{TOK} raw", f"{TOK} +lz4", f"{TOK} +freq", f"{TOK} +ANS",
        f"{TOK} Kalcher(LZMA)", f"{TOK} Kalcher(zstd)",
    ]

    for domain in domains:
        dom_ns = list(ns)
        if args.full:
            test_len = len(load_ids(f"{domain}_test"))
            dom_ns = sorted(set(dom_ns + [test_len // CHUNK_SIZE]))
        results, times = run_domain(domain, dom_ns)

        actual_ns = sorted(results.keys())
        old_n = actual_ns[0]
        print(f"\n{'='*94}")
        print(f"  {domain}: compression ratio vs UTF-8, 512-tok chunks. OLD=N{old_n} vs larger N (o200k for token methods)")
        print(f"{'='*94}")
        hdr = f"  {'method':<22}" + "".join(f"{('N='+str(n)):>12}" for n in actual_ns) + f"{'max %Δ':>10}"
        print(hdr)
        for m in METHODS:
            base = results[old_n][m]
            cells = ""
            max_dp = 0.0
            for n in actual_ns:
                v = results[n][m]
                cells += f"{v:>11.2f}x"
                dp = (v - base) / base * 100
                if abs(dp) > abs(max_dp):
                    max_dp = dp
            print(f"  {m:<22}{cells}{max_dp:>+9.1f}%")
        print(f"\n  wall-clock (ratio compute, excl. one-time setup): " +
              ", ".join(f"N={n}:{times[n]:.1f}s" for n in actual_ns))


if __name__ == "__main__":
    main()
