"""
COLD tokenization latency across tokenizers (r50k, cl100k, o200k).

The paper's serving-latency argument charges the byte path a *cold* tokenize:
each retrieved chunk is a fresh, distinct text the tokenizer sees exactly once.
We already have r50k cold (~283 us, from 07_kalcher_baseline/bench_unified_latency.py's
`timed_once` path); this reproduces r50k and adds cl100k + o200k on the SAME
methodology so the paper can state tokenize cost across tokenizers honestly.

Methodology -- matches bench_unified_latency.py's `timed_once` exactly:
  - ONE Encoding object reused across the whole loop, so the tokenizer internals
    (ranks table, regex) are warm -- only the *text* is cold.
  - Each chunk's enc.encode(text, disallowed_special=()) timed with a SINGLE
    perf_counter shot (cold text, warm tokenizer = the real serving condition).
    NOT the warm timed_reps (30-rep, warm-cache) methodology.
  - Aggregate = median over many DISTINCT chunks (100), bootstrap 90% CI.
  - Single core (RAYON/TIKTOKEN threads = 1). English C4 prose.

Chunk convention:
  - PRIMARY: 512-token chunks in EACH tokenizer's OWN encoding (a stored chunk
    is 512 IDs in that tokenizer) -> the true per-chunk serving cost.
  - Because o200k/cl100k pack more text into 512 tokens, ALSO report us per
    1,000 UTF-8 chars, so the three are comparable apples-to-apples.

Also captures the WARM (timed_reps, 30-rep) number per tokenizer, to state the
cold/warm ratio explicitly.
"""
import os
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TIKTOKEN_MAX_THREADS", "1")

import sys
import json
import time
import numpy as np
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import make_chunks, bootstrap_ci, timed_once, timed_reps, load_ids

CHUNK_SIZE = 512
N_CHUNKS = 100          # distinct cold chunks, more than the 40 baseline to smooth single-shot noise
WARM_REPS = 30
SEED = 3344
TOKENIZERS = {"r50k": "r50k_base", "cl100k": "cl100k_base", "o200k": "o200k_base"}

r50k = tiktoken.get_encoding("r50k_base")


# 64 MB sweep to evict the tokenizer's multi-MB ranks table from CPU cache
# between shots -- reproduces the REAL serving condition where each retrieved
# chunk's tokenize is interleaved with decompression/other work that pollutes
# the cache (this is why the paper's mixed-loop r50k cold was ~283us while a
# back-to-back tokenize loop, ranks-table cache-hot, is far lower).
_POLLUTE = np.ones(8 * 1024 * 1024, dtype=np.int64)


def _cache_pollute():
    _POLLUTE[:] += 1


def timed_once_serving(fn):
    """Cold text AND cold tokenizer-in-cache: evict caches, then single shot."""
    _cache_pollute()
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6


def main():
    rng = np.random.default_rng(SEED)
    test = load_ids("prose_test")

    # Reference r50k 512-token chunks -> canonical English text pieces. To keep
    # "512 tokens in each tokenizer's own encoding", we re-slice each tokenizer's
    # OWN encoding of a large shared text into 512-ID windows (so a stored chunk
    # really is 512 IDs of THAT tokenizer), then decode each window to its text.
    results = {}
    for tok_key, enc_name in TOKENIZERS.items():
        enc = tiktoken.get_encoding(enc_name)  # ONE Encoding object, reused below

        # Build 100 distinct 512-token chunks IN THIS TOKENIZER'S OWN ENCODING.
        # Take distinct r50k windows, decode to text, re-encode with `enc`, and
        # cut the resulting id stream into 512-id windows -> decode back to the
        # exact text of a 512-`enc`-token chunk.
        src_windows = make_chunks(test, CHUNK_SIZE * 4, N_CHUNKS + 20, rng)  # bigger source windows
        texts = []
        for w in src_windows:
            wtext = r50k.decode(w.tolist())
            ids = enc.encode(wtext, disallowed_special=())
            if len(ids) >= CHUNK_SIZE:
                texts.append(enc.decode(ids[:CHUNK_SIZE]))
            if len(texts) >= N_CHUNKS:
                break
        texts = texts[:N_CHUNKS]
        char_lens = np.array([len(t) for t in texts], dtype=np.float64)  # UTF-8 chars (codepoints)
        utf8_lens = np.array([len(t.encode("utf-8")) for t in texts], dtype=np.float64)

        # sanity: each chunk really is 512 ids in this tokenizer
        tok_counts = [len(enc.encode(t, disallowed_special=())) for t in texts]
        assert all(c == CHUNK_SIZE for c in tok_counts), (tok_key, min(tok_counts), max(tok_counts))

        # 512-ID token lists for the decode (detokenize) direction, per chunk
        id_lists = [enc.encode(t, disallowed_special=()) for t in texts]

        # COLD tokenize: single-shot per distinct chunk (text cold, tokenizer warm)
        cold = np.array([timed_once(lambda t=t: enc.encode(t, disallowed_special=())) for t in texts])
        # WARM tokenize: 30-rep median per chunk (warm cache), same chunks
        warm = np.array([timed_reps(lambda t=t: enc.encode(t, disallowed_special=())) for t in texts])
        # COLD detokenize: single-shot decode per distinct chunk (IDs cold, tokenizer warm)
        cold_dec = np.array([timed_once(lambda ids=ids: enc.decode(ids)) for ids in id_lists])
        # WARM detokenize: 30-rep median per chunk
        warm_dec = np.array([timed_reps(lambda ids=ids: enc.decode(ids)) for ids in id_lists])
        # SERVING-COLD: cache evicted before each shot (ranks table cold in CPU
        # cache too) -- matches the paper's mixed-loop condition (~283us r50k).
        cold_srv = np.array([timed_once_serving(lambda t=t: enc.encode(t, disallowed_special=())) for t in texts])
        cold_srv_dec = np.array([timed_once_serving(lambda ids=ids: enc.decode(ids)) for ids in id_lists])

        cold_ci = bootstrap_ci(cold, rng)
        warm_ci = bootstrap_ci(warm, rng)
        cold_dec_ci = bootstrap_ci(cold_dec, rng)
        warm_dec_ci = bootstrap_ci(warm_dec, rng)
        cold_srv_ci = bootstrap_ci(cold_srv, rng)
        cold_srv_dec_ci = bootstrap_ci(cold_srv_dec, rng)
        # per-1k-UTF-8-char normalization: per-chunk cold us / (utf8_bytes/1000)
        cold_per_1k = bootstrap_ci(cold / (utf8_lens / 1000.0), rng)
        cold_dec_per_1k = bootstrap_ci(cold_dec / (utf8_lens / 1000.0), rng)

        results[tok_key] = {
            "cold_tokenize_us_per_chunk_512tok": cold_ci,
            "warm_tokenize_us_per_chunk_512tok": warm_ci,
            "cold_over_warm_ratio_tokenize": cold_ci[0] / warm_ci[0],
            "cold_tokenize_us_per_1k_utf8_chars": cold_per_1k,
            "cold_detokenize_us_per_chunk_512tok": cold_dec_ci,
            "warm_detokenize_us_per_chunk_512tok": warm_dec_ci,
            "cold_over_warm_ratio_detokenize": cold_dec_ci[0] / warm_dec_ci[0],
            "cold_detokenize_us_per_1k_utf8_chars": cold_dec_per_1k,
            "serving_cold_tokenize_us_per_chunk_512tok": cold_srv_ci,
            "serving_cold_detokenize_us_per_chunk_512tok": cold_srv_dec_ci,
            "median_utf8_bytes_per_chunk": float(np.median(utf8_lens)),
            "median_codepoints_per_chunk": float(np.median(char_lens)),
            "n_chunks": len(texts),
        }
        print(
            f"{tok_key:<7} tok_cold={cold_ci[0]:7.1f}us tok_warm={warm_ci[0]:6.1f}us "
            f"(c/w={results[tok_key]['cold_over_warm_ratio_tokenize']:.2f}x)  "
            f"detok_cold={cold_dec_ci[0]:5.1f}us detok_warm={warm_dec_ci[0]:5.1f}us  "
            f"cold_tok/1k-utf8={cold_per_1k[0]:6.1f}us  "
            f"utf8B/chunk={results[tok_key]['median_utf8_bytes_per_chunk']:.0f}",
            flush=True,
        )

    # ── headline labeled block (the unmistakable one) ──
    print("\n=== Cold cost per 512-token chunk (English C4, single core) ===")
    print(f"{'tokenizer':<12}{'tokenize(encode)':<20}{'detokenize(decode)':<20}")
    for tok_key in TOKENIZERS:
        r = results[tok_key]
        tk = f"{r['cold_tokenize_us_per_chunk_512tok'][0]:.1f} us"
        dk = f"{r['cold_detokenize_us_per_chunk_512tok'][0]:.1f} us"
        print(f"{tok_key:<12}{tk:<20}{dk:<20}")
    print("(text cold, tokenizer object + ranks table warm; back-to-back loop)")
    print("\n=== Serving-cold per 512-token chunk (CPU cache evicted between reads) ===")
    print(f"{'tokenizer':<12}{'tokenize(encode)':<20}{'detokenize(decode)':<20}")
    for tok_key in TOKENIZERS:
        r = results[tok_key]
        tk = f"{r['serving_cold_tokenize_us_per_chunk_512tok'][0]:.1f} us"
        dk = f"{r['serving_cold_detokenize_us_per_chunk_512tok'][0]:.1f} us"
        print(f"{tok_key:<12}{tk:<20}{dk:<20}")
    print("(ranks table cold in CPU cache too -- matches the paper's ~283us r50k mixed-loop condition)")

    print(f"\n{'=' * 104}")
    print("COLD vs WARM, English C4 prose, 512-token chunks (each tokenizer's own encoding)")
    print(f"{'=' * 104}")
    print(f"  {'tok':<7}{'tok cold':>10}{'tok warm':>10}{'tok c/w':>9}"
          f"{'detok cold':>12}{'detok warm':>12}{'detok c/w':>11}"
          f"{'cold tok/1k':>13}{'utf8B/chk':>11}")
    for tok_key in TOKENIZERS:
        r = results[tok_key]
        print(f"  {tok_key:<7}"
              f"{r['cold_tokenize_us_per_chunk_512tok'][0]:>9.1f}"
              f"{r['warm_tokenize_us_per_chunk_512tok'][0]:>10.1f}"
              f"{r['cold_over_warm_ratio_tokenize']:>8.2f}x"
              f"{r['cold_detokenize_us_per_chunk_512tok'][0]:>12.1f}"
              f"{r['warm_detokenize_us_per_chunk_512tok'][0]:>12.1f}"
              f"{r['cold_over_warm_ratio_detokenize']:>10.2f}x"
              f"{r['cold_tokenize_us_per_1k_utf8_chars'][0]:>13.1f}"
              f"{r['median_utf8_bytes_per_chunk']:>11.0f}")

    out = {
        "config": {
            "corpus": "prose (English C4)", "chunk_size_tokens": CHUNK_SIZE,
            "n_chunks": N_CHUNKS, "warm_reps": WARM_REPS, "seed": SEED, "single_core": True,
            "cold_method": "timed_once single perf_counter shot per distinct chunk (text cold, "
                           "one reused Encoding object warm) -- matches bench_unified_latency.py",
            "warm_method": "timed_reps median-of-30 per chunk (warm cache)",
            "chunk_convention": "512 IDs in each tokenizer's OWN encoding; also normalized per 1k UTF-8 bytes",
        },
        "tokenizers": results,
    }
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
