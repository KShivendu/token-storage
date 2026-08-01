# Tokenize / detokenize latency — rules of thumb

Quick reference for the cost of the byte-storage path's mandatory tokenize (on read)
and detokenize (on write). Token-native storage pays **neither** — the data is already
token IDs. All numbers: **per 512-token chunk, single core, `tiktoken`**, English C4,
measured across r50k / cl100k / o200k (they land within ~1.2× of each other).

Source of truth: [`09_cold_tokenize/results.json`](09_cold_tokenize/results.json)
(run `09_cold_tokenize/bench_cold_tokenize.py` to reproduce; it prints a labeled block).

## The table

| operation | warm | serving-cold |
|---|--:|--:|
| **detokenize** (decode → text) | ~10 µs | ~45 µs |
| **tokenize** (encode → IDs) | ~100 µs | ~230 µs |

Per-tokenizer serving-cold tokenize: r50k 208, cl100k 246, o200k 235 µs (synthetic
cache-evictor). The real interleaved serving workload
(`07_kalcher_baseline/bench_unified_latency.py`) pushes r50k to **~283 µs** — treat
**~210–280 µs** as the honest serving-cold band.

## Rules of thumb

1. **Tokenize ≈ 10× detokenize.** Encoding is the expensive half; the *read* path
   (which must tokenize) is the pole, not the write path.
2. **~90 µs per 1 KB of UTF-8 text to tokenize (serving-cold);** ~50 µs/KB warm. Cost
   tracks *text length in bytes*, and is roughly tokenizer-independent.
3. **A 512-token English chunk ≈ 2.3 KB** (2,344 B r50k / 2,406 cl100k / 2,438 o200k).
   Handy conversions: ~0.75 words/token (≈4/3 tokens per word), ~4 chars/token,
   ~4.6 bytes/token. (512 tokens ≈ 384 words ≈ 2.3 KB — *not* 4 KB; don't invert the
   tokens/word ratio.)
4. **Serving-cold ≈ 2× warm** for tokenize; ~3–4× for detokenize (off a small base).
   The lever is **CPU-cache residency of the tokenizer's multi-MB rank table**, not
   data warmth (repeating the same text barely helps: 106 vs 116 µs for r50k).
5. **Budget with serving-cold, not warm.** "Warm" only happens in a tight benchmark
   loop that keeps the rank table resident. Real serving interleaves codec/model work
   that evicts it, so every read reloads cold.
6. **In serving reality there is no "warm tokenize," but "warm detokenize" is real.**
   Read-time tokenize is sporadic and interleaved → always table-cold (~230 µs).
   Generation-time detokenize runs in the per-token streaming loop → table stays hot
   → genuine ~10 µs. So the byte path's expensive op (cold tokenize on read) has no
   warm relief, while the cheap op (detokenize) is warm anyway.

## Two "cold" definitions (they differ ~2×)

- **text-cold** — fresh text, but a back-to-back loop keeps the rank table
  cache-resident (≈ batch/offline tokenization). r50k tokenize ~116 µs.
- **serving-cold** — the rank table is evicted between ops by interleaved work
  (≈ latency-sensitive serving). r50k tokenize ~208–283 µs.

Use **serving-cold** for the paper's agent-latency argument; it's the realistic
condition for reads of distinct chunks.

## Why this matters

The asymmetry — **read ≫ write, tokenize ≫ detokenize, cold tokenize with no warm
path** — is exactly why token-native storage's advantage is largest on
retrieval-heavy, read-latency-bound agent loops.
