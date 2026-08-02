# Tokenize / detokenize latency — rules of thumb

Quick reference for the cost of the byte-storage path's mandatory tokenize (on read)
and detokenize (on write). Token-native storage pays **neither** — the data is already
token IDs. All numbers: **per 512-token chunk, single core, `tiktoken`**, English C4,
measured across r50k / cl100k / o200k (they land within ~1.2× of each other).
Always run pinned to a P-core — see the pinning section below.

Source of truth: [`09_cold_tokenize/results.json`](09_cold_tokenize/results.json)
(run `09_cold_tokenize/bench_cold_tokenize.py` to reproduce; it prints a labeled block).

## The table

| operation | warm | serving-cold |
|---|--:|--:|
| **detokenize** (decode → text) | ~10 µs | ~45 µs |
| **tokenize** (encode → IDs) | ~130 µs | ~250 µs |

Per-tokenizer serving-cold tokenize: **r50k 248, cl100k 291, o200k 267 µs**
(`09_cold_tokenize`, synthetic cache-evictor, P-core pinned). Two independent
harnesses agree: `03_latency`'s grid gets r50k 240 µs with the same evictor, and
`07_kalcher_baseline`'s interleaved loop gets 283 µs without one. Treat
**~240–283 µs** as the serving-cold band for r50k, and quote a band rather than a
point when it matters.

Serving-cold detokenize: **r50k 45, cl100k 44–51, o200k 51–56 µs** across the two
harnesses.

## Rules of thumb

1. **Tokenize ≈ 10× detokenize.** Encoding is the expensive half; the *read* path
   (which must tokenize) is the pole, not the write path.
2. **~110 µs per 1 KB of UTF-8 text to tokenize (serving-cold),** ~55 µs/KB text-cold.
   Cost tracks *text length in bytes*, and is roughly tokenizer-independent.
3. **A 512-token English chunk ≈ 2.3 KB** (2,344 B r50k / 2,406 cl100k / 2,438 o200k).
   Handy conversions: ~0.75 words/token (≈4/3 tokens per word), ~4 chars/token,
   ~4.6 bytes/token. (512 tokens ≈ 384 words ≈ 2.3 KB — *not* 4 KB; don't invert the
   tokens/word ratio.)
4. **Serving-cold ≈ 2× warm** for tokenize, ~3–4× for detokenize (off a small base).
   The lever is **CPU-cache residency of the tokenizer's multi-MB rank table**, not
   data warmth (repeating the same text barely helps: 134 vs 149 µs for r50k).
5. **Budget with serving-cold, not warm.** "Warm" only happens in a tight benchmark
   loop that keeps the rank table resident. Real serving interleaves codec/model work
   that evicts it, so every read reloads cold.
6. **In serving reality there is no "warm tokenize," but "warm detokenize" is real.**
   Read-time tokenize is sporadic and interleaved → always table-cold (~250 µs).
   Generation-time detokenize runs in the per-token streaming loop → table stays hot
   → genuine ~10 µs. So the byte path's expensive op (cold tokenize on read) has no
   warm relief, while the cheap op (detokenize) is warm anyway.

## Two "cold" definitions (they differ ~2×)

- **text-cold** — fresh text, but a back-to-back loop keeps the rank table
  cache-resident (≈ batch/offline tokenization). r50k tokenize ~116 µs.
- **serving-cold** — the rank table is evicted between ops by interleaved work
  (≈ latency-sensitive serving). r50k tokenize ~208–283 µs.

Use **serving-cold** for the paper's agent-latency argument. It is the realistic
condition for reads of distinct chunks, and it is what every bench in this repo
now measures: the timer is `tnbench.timed_once_serving` (64 MB cache sweep before
each single shot) and both `09_cold_tokenize` and `03_latency` call it.

## Pin to a P-core, or the numbers swing 1.6×

These are single-core benchmarks and this dev box is a hybrid CPU (Core Ultra 7
155H: P-cores at 4.8 GHz, LP-E cores at 2.5 GHz). An unpinned run can be migrated
onto an LP-E core mid-run, which inflates every cell by ~1.6× and is the main
source of run-to-run disagreement between committed result files. Always run:

```bash
taskset -c 4 env RAYON_NUM_THREADS=1 TIKTOKEN_MAX_THREADS=1 uv run python <bench>.py
```

Check `cat /sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq` first and pick a
core at the top frequency. Avoid cpu0, which handles interrupts.

## Why this matters

The asymmetry — **read ≫ write, tokenize ≫ detokenize, cold tokenize with no warm
path** — is exactly why token-native storage's advantage is largest on
retrieval-heavy, read-latency-bound agent loops.
