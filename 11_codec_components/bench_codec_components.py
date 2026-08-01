"""
Byte-storage path, decomposed: compress / decompress latency for every byte
codec on REAL English C4 chunks, then composed into full agent read/write so we
can show honestly that the byte codecs which MATCH token-native's ratio
(zstd-19, brotli-q11) are slow to WRITE.

For each byte codec (LZ4, gzip-9, zstd-19, brotli-q11, zstd --train) on 512-token
C4 chunks (single core, median over >=40 chunks):
  - compress us, decompress us  (warm / median-of-30, matching
    bench_unified_latency.py's convention for the small deterministic codec
    halves)
  - compressed bytes + ratio vs raw UTF-8 (real chunks, not synthetic)

Then compose the full agent path using the SERVING-COLD tokenize / detokenize
from 09_cold_tokenize (r50k, cache-evicted single shot -- the real serving
condition where each fresh chunk's tokenize is interleaved with other work):
  byte WRITE = detokenize + compress
  byte READ  = decompress + tokenize

Token-native write (0.6-4.4us) and read (0.4-30us) from
07_kalcher_baseline/bench_unified_latency.py are shown as reference rows.
"""
import os
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TIKTOKEN_MAX_THREADS", "1")

import json
import time
import gzip
import numpy as np
import tiktoken
import lz4.frame as lz4f
import zstandard as zstd
import brotli

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
CHUNK_SIZE = 512
N_CHUNKS = 50
REPS = 30
N_BOOTSTRAP = 2000
SEED = 3344

r50k = tiktoken.get_encoding("r50k_base")
zstd_c19 = zstd.ZstdCompressor(level=19)
zstd_d = zstd.ZstdDecompressor()

# token-native reference rows (English, from bench_unified_latency.py)
TOKEN_NATIVE_REF = {
    "r50k raw":  {"write": 0.6, "read": 0.4},
    "r50k +freq": {"write": 3.2, "read": 4.0},
    "r50k +ANS":  {"write": 4.4, "read": 28.6},
}


def bootstrap_ci(values, rng, n_boot=N_BOOTSTRAP, alpha=0.10):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(np.median(values)), (float(values[0]), float(values[0]))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    bm = np.median(values[idx], axis=1)
    lo, hi = np.percentile(bm, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(values)), (float(lo), float(hi))


def make_chunks(arr, chunk_size, n_chunks, rng):
    max_chunks = len(arr) // chunk_size
    n = min(n_chunks, max_chunks)
    starts = rng.choice(max_chunks, size=n, replace=False) * chunk_size
    return [arr[s: s + chunk_size] for s in starts]


def timed_reps(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e6)
    return float(np.median(ts))


# serving-cold: evict CPU cache (64MB sweep) before a single-shot tokenize/detok
_POLLUTE = np.ones(8 * 1024 * 1024, dtype=np.int64)


def timed_once_serving(fn):
    _POLLUTE[:] += 1
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6


def main():
    rng = np.random.default_rng(SEED)
    test = np.load(os.path.join(CORPUS_DIR, "prose_test.npy")).astype(np.int64)
    train = np.load(os.path.join(CORPUS_DIR, "prose_train.npy")).astype(np.int64)

    chunks = make_chunks(test, CHUNK_SIZE, N_CHUNKS, rng)
    texts = [r50k.decode(c.tolist()) for c in chunks]
    raws = [t.encode("utf-8") for t in texts]
    id_lists = [c.astype(np.int64).tolist() for c in chunks]

    # ── serving-cold tokenize / detokenize (r50k), the composition inputs ──
    tok_cold = np.array([timed_once_serving(lambda t=t: r50k.encode(t, disallowed_special=())) for t in texts])
    detok_cold = np.array([timed_once_serving(lambda ids=ids: r50k.decode(ids)) for ids in id_lists])
    tok_ci = bootstrap_ci(tok_cold, rng)
    detok_ci = bootstrap_ci(detok_cold, rng)
    print(f"serving-cold tokenize={tok_ci[0]:.1f}us  detokenize={detok_ci[0]:.1f}us", flush=True)

    # ── zstd --train dict (112K, trained on train split chunks) ──
    train_samples = [r50k.decode(train[i:i + CHUNK_SIZE].tolist()).encode("utf-8")
                     for i in range(0, min(len(train), CHUNK_SIZE * 400), CHUNK_SIZE)]
    zdict = zstd.ZstdCompressionDict(zstd.train_dictionary(112 * 1024, train_samples).as_bytes())
    zc_dict = zstd.ZstdCompressor(level=19, dict_data=zdict)
    zd_dict = zstd.ZstdDecompressor(dict_data=zdict)

    codecs = {
        "LZ4":          (lambda b: lz4f.compress(b),           lambda c: lz4f.decompress(c)),
        "gzip-9":       (lambda b: gzip.compress(b, 9),        lambda c: gzip.decompress(c)),
        "zstd-19":      (lambda b: zstd_c19.compress(b),       lambda c: zstd_d.decompress(c)),
        "brotli-q11":   (lambda b: brotli.compress(b, quality=11), lambda c: brotli.decompress(c)),
        "zstd --train": (lambda b: zc_dict.compress(b),        lambda c: zd_dict.decompress(c)),
    }

    results = {
        "config": {
            "corpus": "prose (English C4)", "chunk_size": CHUNK_SIZE, "n_chunks": N_CHUNKS,
            "reps": REPS, "seed": SEED, "single_core": True,
            "codec_timing": "warm median-of-30 (matches bench_unified_latency.py codec halves)",
            "tokenize_timing": "serving-cold single-shot, cache-evicted (matches 09_cold_tokenize)",
            "compose": "byte write = detokenize + compress ; byte read = decompress + tokenize",
        },
        "serving_cold_tokenize_us": tok_ci,
        "serving_cold_detokenize_us": detok_ci,
        "codecs": {},
        "token_native_reference": TOKEN_NATIVE_REF,
    }

    for name, (comp, decomp) in codecs.items():
        comp_t, dec_t, ratios, cbytes = [], [], [], []
        for raw in raws:
            c = comp(raw)
            assert decomp(c) == raw
            comp_t.append(timed_reps(lambda r=raw: comp(r)))
            dec_t.append(timed_reps(lambda c=c: decomp(c)))
            ratios.append(len(raw) / len(c))
            cbytes.append(len(c))
        comp_ci = bootstrap_ci(comp_t, rng)
        dec_ci = bootstrap_ci(dec_t, rng)
        ratio_ci = bootstrap_ci(ratios, rng)
        full_write = detok_ci[0] + comp_ci[0]     # detokenize + compress
        full_read = dec_ci[0] + tok_ci[0]         # decompress + tokenize
        results["codecs"][name] = {
            "compress_us": comp_ci,
            "decompress_us": dec_ci,
            "ratio": ratio_ci,
            "median_compressed_bytes": float(np.median(cbytes)),
            "full_byte_write_us": full_write,
            "full_byte_read_us": full_read,
        }
        print(f"  {name:<13} compress={comp_ci[0]:8.1f}us  decompress={dec_ci[0]:6.1f}us  "
              f"ratio={ratio_ci[0]:.2f}x  write={full_write:8.1f}us  read={full_read:7.1f}us", flush=True)

    # ── summary table ──
    print(f"\n{'=' * 100}")
    print("BYTE CODEC COMPONENTS -- English C4, 512-tok chunks, single core, median us/chunk")
    print(f"  (byte write = detok {detok_ci[0]:.0f}us + compress ; byte read = decompress + tokenize {tok_ci[0]:.0f}us)")
    print(f"{'=' * 100}")
    print(f"  {'codec':<14}{'compress':>11}{'decompress':>12}{'ratio':>8}{'bytes':>8}"
          f"{'FULL WRITE':>13}{'FULL READ':>12}")
    for name, m in results["codecs"].items():
        print(f"  {name:<14}{m['compress_us'][0]:>10.1f}{m['decompress_us'][0]:>12.1f}"
              f"{m['ratio'][0]:>7.2f}x{m['median_compressed_bytes']:>8.0f}"
              f"{m['full_byte_write_us']:>12.1f}{m['full_byte_read_us']:>12.1f}")
    print("  --- token-native reference (bench_unified_latency.py) ---")
    for name, m in TOKEN_NATIVE_REF.items():
        print(f"  {name:<14}{'-':>10}{'-':>12}{'-':>8}{'-':>8}{m['write']:>12.1f}{m['read']:>12.1f}")

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
