"""
~1GB-scale compression-ratio measurement (enwik9 parity with Kalcher).

Same pipeline and same per-chunk computations as bench_ratio_scaleup.py (byte
codecs on the r50k-decoded UTF-8 text; token-native methods at o200k), but:

  * reads a configurable corpus dir (data/corpus_1gb),
  * splits methods into a FAST tier (run on EVERY 512-token test chunk) and a
    SLOW tier (brotli-q11, Kalcher(LZMA)) run on a capped random sub-sample,
    because brotli-q11 + LZMA-extreme on ~1GB is infeasible on 2 cores in a
    couple hours -- and the median ratio has long since converged by tens of
    thousands of chunks (the paper's own N=40 -> 1.5M check moves <2%),
  * parallelizes the per-chunk loop across cores (fork; workers inherit the
    memmapped test array copy-on-write and rebuild codec state once each).

`--calibrate M` times every method on M chunks (single core) and projects the
full-N wall clock, then exits -- run this first to size the slow cap sanely.

Median ratios are identical in expectation to the serial version; only N and the
execution order differ. N actually used is printed per method.

Usage:
  uv run python 01_storage_efficiency/bench_ratio_1gb.py \
      --corpus-dir data/corpus_1gb --domains prose,code,hindi \
      [--slow-cap 50000] [--workers 2] [--calibrate 400]
"""
import argparse
import gzip
import lzma
import os
import sys
import time
import multiprocessing as mp

import brotli
import lz4.frame
import numpy as np
import tiktoken
import zstandard as zstd
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import (
    make_chunks, build_rank_table, leb128_encode, svb_encode_arr,
    LZMA_FILTERS, pack3,
)

CHUNK_SIZE = 512
SEED = 9012
TOK = "o200k"
TOK_ENC_NAME = "o200k_base"
VOCAB = 200019

FAST_METHODS = [
    "LZ4", "gzip-9", "zstd-19", "zstd --train",
    f"{TOK} raw", f"{TOK} +lz4", f"{TOK} +freq", f"{TOK} +ANS",
    f"{TOK} Kalcher(zstd)",
]
SLOW_METHODS = ["brotli-q11", f"{TOK} Kalcher(LZMA)"]
ALL_METHODS = [
    "LZ4", "gzip-9", "zstd-19", "brotli-q11", "zstd --train",
    f"{TOK} raw", f"{TOK} +lz4", f"{TOK} +freq", f"{TOK} +ANS",
    f"{TOK} Kalcher(LZMA)", f"{TOK} Kalcher(zstd)",
]

# ── per-worker state (built once per process in _init) ───────────────────────
_G = {}


def _init(corpus_dir, domain, rank_of, ans_probs, zdict_bytes):
    r50k = tiktoken.get_encoding("r50k_base")
    enc = tiktoken.get_encoding(TOK_ENC_NAME)
    zdict = zstd.ZstdCompressionDict(zdict_bytes)
    zdict.precompute_compress(level=19)
    _G.update(
        test=np.load(os.path.join(corpus_dir, f"{domain}_test.npy"), mmap_mode="r"),
        r50k=r50k,
        enc=enc,
        rank_of=rank_of,
        model=constriction.stream.model.Categorical(ans_probs, perfect=False),
        zc19=zstd.ZstdCompressor(level=19),
        zc22=zstd.ZstdCompressor(level=22),
        zc_dict=zstd.ZstdCompressor(level=19, dict_data=zdict),
    )


def _chunk_text_and_ids(start):
    g = _G
    chunk = np.asarray(g["test"][start:start + CHUNK_SIZE]).astype(np.int64)
    text = g["r50k"].decode(chunk.tolist())
    raw = text.encode("utf-8")
    ids = np.array(g["enc"].encode(text, disallowed_special=()), dtype=np.int64)
    return raw, ids


def work_fast(start):
    g = _G
    raw, ids = _chunk_text_and_ids(start)
    rl = len(raw)
    packed = pack3(ids)
    remapped = g["rank_of"][ids]
    c = constriction.stream.stack.AnsCoder()
    c.encode_reverse(ids.astype(np.int32), g["model"])
    varint = leb128_encode(remapped)
    return (
        rl / len(lz4.frame.compress(raw)),
        rl / len(gzip.compress(raw, compresslevel=9)),
        rl / len(g["zc19"].compress(raw)),
        rl / len(g["zc_dict"].compress(raw)),
        rl / len(packed),
        rl / len(lz4.frame.compress(packed)),
        rl / len(svb_encode_arr(remapped)),
        rl / len(c.get_compressed().tobytes()),
        rl / len(g["zc22"].compress(varint)),
    )


def work_slow(start):
    g = _G
    raw, ids = _chunk_text_and_ids(start)
    rl = len(raw)
    remapped = g["rank_of"][ids]
    varint = leb128_encode(remapped)
    return (
        rl / len(brotli.compress(raw, quality=11)),
        rl / len(lzma.compress(varint, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)),
    )


def work_all(start):
    """Serial calibration: every method on one chunk."""
    return work_fast(start), work_slow(start)


def setup_domain(corpus_dir, domain):
    """Parent-side one-time setup: rank table, ANS probs, zstd-dict bytes."""
    t0 = time.perf_counter()
    r50k = tiktoken.get_encoding("r50k_base")
    enc = tiktoken.get_encoding(TOK_ENC_NAME)
    train = np.load(os.path.join(corpus_dir, f"{domain}_train.npy"))
    train_text = r50k.decode(train.tolist())
    train_ids = enc.encode(train_text, disallowed_special=())
    rank_of, _ = build_rank_table(train_ids, VOCAB)
    counts = np.ones(VOCAB, dtype=np.int64)
    counts += np.bincount(np.asarray(train_ids, dtype=np.int64), minlength=VOCAB)
    ans_probs = counts.astype(np.float64) / counts.sum()
    # zstd --train dict over full train split, r50k-decoded 512-token windows.
    samples = [
        r50k.decode(train[i:i + CHUNK_SIZE].tolist()).encode("utf-8")
        for i in range(0, (len(train) // CHUNK_SIZE) * CHUNK_SIZE, CHUNK_SIZE)
    ]
    zdict_bytes = zstd.train_dictionary(112 * 1024, samples).as_bytes()
    print(f"  setup (o200k train encode + rank/ANS + zstd-dict): {time.perf_counter()-t0:.1f}s", flush=True)
    return rank_of, ans_probs, zdict_bytes


def _starts(corpus_dir, domain, n=None):
    test = np.load(os.path.join(corpus_dir, f"{domain}_test.npy"), mmap_mode="r")
    max_chunks = len(test) // CHUNK_SIZE
    rng = np.random.default_rng(SEED)
    k = max_chunks if n is None else min(n, max_chunks)
    return (rng.choice(max_chunks, size=k, replace=False) * CHUNK_SIZE).astype(np.int64), max_chunks


def run_domain(corpus_dir, domain, slow_cap, workers, calibrate):
    print(f"\n=== {domain} ===", flush=True)
    rank_of, ans_probs, zdict_bytes = setup_domain(corpus_dir, domain)
    ctx = mp.get_context("fork")
    initargs = (corpus_dir, domain, rank_of, ans_probs, zdict_bytes)

    if calibrate:
        starts, maxc = _starts(corpus_dir, domain, calibrate)
        _init(*initargs)
        ss = [int(s) for s in starts]
        t0 = time.perf_counter()
        for s in ss:
            work_fast(s)
        per_f = (time.perf_counter() - t0) / len(ss)
        t0 = time.perf_counter()
        for s in ss:
            work_slow(s)
        per_s = (time.perf_counter() - t0) / len(ss)
        print(f"  calibrate N={len(ss)} single-core:", flush=True)
        print(f"    fast tier {per_f*1000:.2f} ms/chunk, slow tier {per_s*1000:.2f} ms/chunk", flush=True)
        print(f"  max test chunks available: {maxc:,}", flush=True)
        print(f"  projected FULL fast pass @ {workers}w over {maxc:,}: {maxc*per_f/workers/60:.1f} min", flush=True)
        print(f"  projected SLOW pass @ {workers}w over {slow_cap:,}: {slow_cap*per_s/workers/60:.1f} min", flush=True)
        return None

    # FAST pass over ALL chunks
    starts_all, maxc = _starts(corpus_dir, domain, None)
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init, initargs=initargs) as pool:
        fast = np.array(pool.map(work_fast, [int(s) for s in starts_all], chunksize=256))
    t_fast = time.perf_counter() - t0
    fast_med = {m: float(np.median(fast[:, i])) for i, m in enumerate(FAST_METHODS)}
    n_fast = len(starts_all)
    print(f"  FAST pass: N={n_fast:,} chunks in {t_fast/60:.1f} min", flush=True)

    # SLOW pass over capped sub-sample
    starts_slow, _ = _starts(corpus_dir, domain, slow_cap)
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init, initargs=initargs) as pool:
        slow = np.array(pool.map(work_slow, [int(s) for s in starts_slow], chunksize=64))
    t_slow = time.perf_counter() - t0
    slow_med = {m: float(np.median(slow[:, i])) for i, m in enumerate(SLOW_METHODS)}
    n_slow = len(starts_slow)
    print(f"  SLOW pass: N={n_slow:,} chunks in {t_slow/60:.1f} min", flush=True)

    med = {**fast_med, **slow_med}
    n_used = {m: n_fast for m in FAST_METHODS}
    n_used.update({m: n_slow for m in SLOW_METHODS})
    return med, n_used, {"fast_min": t_fast / 60, "slow_min": t_slow / 60, "max_chunks": maxc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="data/corpus_1gb")
    ap.add_argument("--domains", default="prose,code,hindi")
    ap.add_argument("--slow-cap", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--calibrate", type=int, default=0)
    args = ap.parse_args()
    corpus_dir = os.path.abspath(args.corpus_dir)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    OLD = {  # 1.5M-scale headline ratios (o200k, from the task brief / Table 1), prose only known
        "prose": {f"{TOK} raw": 1.59, f"{TOK} +freq": 2.73, f"{TOK} +ANS": 3.40,
                  f"{TOK} Kalcher(zstd)": 3.22, "LZ4": 1.27, "gzip-9": 1.92,
                  "zstd-19": 1.94, "brotli-q11": 2.57, "zstd --train": 2.72},
    }

    for domain in domains:
        out = run_domain(corpus_dir, domain, args.slow_cap, args.workers, args.calibrate)
        if out is None:
            continue
        med, n_used, timing = out
        print(f"\n{'='*78}")
        print(f"  {domain}: NEW ~1GB-scale median compression ratio vs UTF-8 (512-tok chunks)")
        print(f"  fast pass {timing['fast_min']:.1f} min, slow pass {timing['slow_min']:.1f} min, "
              f"max_chunks={timing['max_chunks']:,}")
        print(f"{'='*78}")
        old = OLD.get(domain, {})
        print(f"  {'method':<22}{'old(1.5M)':>12}{'new(1GB)':>12}{'N_used':>12}{'Δ%':>8}")
        for m in ALL_METHODS:
            nv = med[m]
            ov = old.get(m)
            ostr = f"{ov:.2f}x" if ov is not None else "   --"
            dstr = f"{(nv-ov)/ov*100:+.1f}%" if ov is not None else "   --"
            print(f"  {m:<22}{ostr:>12}{nv:>11.2f}x{n_used[m]:>12,}{dstr:>8}")


if __name__ == "__main__":
    main()
