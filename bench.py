#!/usr/bin/env python
"""Thin, reproducible CLI over the token-storage benchmark suite.

This is ONLY a config + dispatch layer. Every measurement it reports comes from
functions that already live in the repo:

  - ratio / latency / chunk-sweep -> 03_latency/bench_latency_grid.py's run_cell
    and byte_codec_components (agent read/write latency + ratio per method), the
    same functions the committed Table 2 and the chunk-size appendix use.
  - block2x2 -> the --codec-blocks / --token-blocks / --read-2x2 flags of
    07_kalcher_baseline/bench_kalcher_table1_matched.py (byte/token x
    document/block ratio and single-document read latency).

No measurement logic is defined here. Tokenizer selection is always a list mapped
per corpus at run time, so a cross-corpus run can never silently fall back to one
hardcoded tokenizer (the r50k-on-Hindi understatement is impossible here).

Run pinned for latency, exactly as the underlying benches document:
  taskset -c 4 env RAYON_NUM_THREADS=1 TIKTOKEN_MAX_THREADS=1 \
    uv run python bench.py --experiment latency --chunk-sizes 512
"""
import argparse
import json
import os
import subprocess
import sys

import tiktoken

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "03_latency"))
sys.path.insert(0, REPO)

import bench_latency_grid as blg  # noqa: E402  (reused measurement functions)
from tnbench import seeded_rng  # noqa: E402

ENC_NAME = {"r50k": "r50k_base", "cl100k": "cl100k_base", "o200k": "o200k_base"}
VOCAB = {"r50k": 50257, "cl100k": 100277, "o200k": 200019}
ALL_TOKENIZERS = list(ENC_NAME)
ALL_CORPORA = ["prose", "code", "hindi"]
TOKEN_METHODS = list(blg.TOKEN_METHODS)          # raw, +freq, +ANS, +dict, Kalcher..., dict-freqvarint
BYTE_CODECS = list(blg.BYTE_CODECS)              # LZ4, gzip-9, zstd-19, brotli-q11, zstd --train
ALL_METHODS = BYTE_CODECS + TOKEN_METHODS


def collect(tokenizers, corpora, chunk_sizes, seed, n_chunks):
    """Loop the full requested cross product, reusing run_cell + byte_codec_components.
    Byte codecs are tokenizer-independent for ratio/compress/decompress, so they are
    measured once per (corpus, chunk_size) and reused; their agent read/write add the
    per-tokenizer serving-cold (de)tokenize tax. Returns {(tok, corpus, cs, method):
    {ratio, read_us, write_us, kind}}."""
    blg.N_CHUNKS = n_chunks
    blg.SEED = seed
    enc_cache = {}
    res = {}
    for cs in chunk_sizes:
        blg.CHUNK_SIZE = cs
        for domain in corpora:
            di = ALL_CORPORA.index(domain)  # full index -> chunk selection matches full runs
            byte = None
            for tok in tokenizers:
                enc = enc_cache.setdefault(tok, tiktoken.get_encoding(ENC_NAME[tok]))
                rng = seeded_rng(seed, cs, di)  # identical across tokenizers -> same chunks
                cell, texts, train = blg.run_cell(domain, tok, enc, VOCAB[tok], tok == "r50k", rng)
                if byte is None:
                    byte = blg.byte_codec_components(texts, train, rng)
                tc = cell["tokenize_serving_cold_us"][0]
                dc = cell["detokenize_serving_cold_us"][0]
                for m in TOKEN_METHODS:
                    mm = cell["methods"][m]
                    res[(tok, domain, cs, m)] = {"kind": "token", "ratio": mm["ratio"][0],
                                                 "read_us": mm["read_us"][0], "write_us": mm["write_us"][0]}
                for name in BYTE_CODECS:
                    b = byte[name]
                    res[(tok, domain, cs, name)] = {"kind": "byte", "ratio": b["ratio"][0],
                                                    "read_us": b["decompress_us"][0] + tc,
                                                    "write_us": b["compress_us"][0] + dc}
                print(f"  done: {tok} / {domain} / cs={cs}", flush=True)
    return res


def print_field(res, tokenizers, corpora, chunk_sizes, methods, field, label, fmt):
    print(f"\n{'#' * 88}\n  {label}  (cells = {'/'.join(corpora)})\n{'#' * 88}")
    for tok in tokenizers:
        print(f"\n  --- tokenizer = {tok} ---")
        print(f"  {'method':<13}{'kind':<6}" + "".join(f"{str(cs):>20}" for cs in chunk_sizes))
        for m in methods:
            kind = next((res[(tok, d, cs, m)]["kind"] for cs in chunk_sizes for d in corpora
                         if (tok, d, cs, m) in res), "?")
            row = f"  {m:<13}{kind:<6}"
            for cs in chunk_sizes:
                cells = [fmt.format(res[(tok, d, cs, m)][field]) for d in corpora if (tok, d, cs, m) in res]
                row += ("/".join(cells)).rjust(20)
            print(row)


def run_measured(args, experiment):
    tokenizers = args.tokenizers
    corpora = args.corpora
    chunk_sizes = args.chunk_sizes
    methods = [m for m in ALL_METHODS if m in args.methods]
    res = collect(tokenizers, corpora, chunk_sizes, args.seed, args.n_chunks)

    if experiment in ("ratio", "chunk-sweep"):
        print_field(res, tokenizers, corpora, chunk_sizes, methods, "ratio", "RATIO (median x vs raw UTF-8)", "{:.2f}")
    if experiment in ("latency", "chunk-sweep"):
        print_field(res, tokenizers, corpora, chunk_sizes, methods, "read_us", "AGENT READ latency (us)", "{:.0f}")
        print_field(res, tokenizers, corpora, chunk_sizes, methods, "write_us", "AGENT WRITE latency (us)", "{:.0f}")

    out = {"config": {"experiment": experiment, "tokenizers": tokenizers, "corpora": corpora,
                      "chunk_sizes": chunk_sizes, "methods": methods, "seed": args.seed, "n_chunks": args.n_chunks},
           "results": {f"{t}|{d}|{cs}|{m}": v for (t, d, cs, m), v in res.items()}}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.out}")


def run_block2x2(args):
    """Dispatch to the existing block/2x2 flags. That script carries its own fixed
    config (512-tok, ES/Lucene block sizes, script-appropriate tokenizer per corpus),
    so the generic knobs do not apply here; we just run the three modes."""
    script = os.path.join(REPO, "07_kalcher_baseline", "bench_kalcher_table1_matched.py")
    for flag in ["--codec-blocks", "--token-blocks", "--read-2x2"]:
        print(f"\n===== {script} {flag} =====", flush=True)
        subprocess.run([sys.executable, script, flag], check=True)


def main():
    p = argparse.ArgumentParser(
        prog="bench.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Thin config + dispatch CLI over the token-storage benchmarks (reuses existing "
                    "run_cell / byte_codec_components / block flags; defines no measurement logic).")
    p.add_argument("--experiment", required=True, choices=["ratio", "latency", "block2x2", "chunk-sweep"],
                   help="ratio=Table 1 compression; latency=Table 2 agent read/write; "
                        "block2x2=byte/token x document/block; chunk-sweep=ratio+latency across sizes.")
    p.add_argument("--tokenizers", nargs="+", default=ALL_TOKENIZERS, choices=ALL_TOKENIZERS,
                   help="Subset of tokenizers; always a list mapped per corpus (never one hardcoded).")
    p.add_argument("--corpora", nargs="+", default=ALL_CORPORA, choices=ALL_CORPORA,
                   help="Subset of corpora.")
    p.add_argument("--chunk-sizes", nargs="+", type=int, default=[512],
                   help="Chunk sizes in tokens (chunk-sweep default sweeps 512 1024 2048 4096).")
    p.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS,
                   help="Subset of methods (byte codecs and/or token-native).")
    p.add_argument("--seed", type=int, default=9012, help="RNG seed for chunk selection.")
    p.add_argument("--n-chunks", type=int, default=40, help="Sampled test chunks per corpus.")
    p.add_argument("--out", default=None, help="Output JSON path (default results_<experiment>.json).")
    args = p.parse_args()

    if args.experiment == "chunk-sweep" and args.chunk_sizes == [512]:
        args.chunk_sizes = [512, 1024, 2048, 4096]
    if args.out is None:
        args.out = os.path.join(REPO, f"results_{args.experiment}.json")

    print("Resolved config:")
    print(f"  experiment  = {args.experiment}")
    print(f"  tokenizers  = {args.tokenizers}")
    print(f"  corpora     = {args.corpora}")
    print(f"  chunk_sizes = {args.chunk_sizes}")
    print(f"  methods     = {[m for m in ALL_METHODS if m in args.methods]}")
    print(f"  seed        = {args.seed}   n_chunks = {args.n_chunks}")
    print(f"  out         = {args.out if args.experiment != 'block2x2' else '(block2x2 writes its own *_results.json)'}")

    if args.experiment == "block2x2":
        run_block2x2(args)
    else:
        run_measured(args, args.experiment)


if __name__ == "__main__":
    main()
