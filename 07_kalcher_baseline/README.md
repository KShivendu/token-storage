# 07 — Kalcher baseline

Adds **Kalcher 2026** ("Frequency-Ordered Tokenization for Better Text
Compression", arXiv:2602.22958) as a baseline and benchmarks it head-to-head
against this repo's token-native methods, on the same corpora, 512-token
chunks, train/test split, bootstrap CIs, and median-of-30-reps latency as
`03_latency/` and `04_frequency_remap/`.

Run: `uv run python 07_kalcher_baseline/bench_kalcher.py` → `results.json`.

## Kalcher's pipeline (as implemented)

1. BPE tokenize — reuse the repo's r50k token-id arrays.
2. Frequency-reorder the vocab (most frequent token → smallest ID). This is the
   **same rank table `+freq` already builds** (argsort desc frequency on the
   train split), so we reuse it.
3. **LEB128 varint** encode the remapped IDs (7 value bits + 1 continuation bit
   per byte; 1/2/3 bytes for ranks 0–127 / 128–16383 / 16384–2097151).
   Implemented here (`leb128_encode`/`leb128_decode`, vectorized in numpy so we
   don't strawman it with a slow pure-Python loop) — **not** streamvbyte.
4. General-purpose compressor on the varint stream. Kalcher's Table I tested
   zlib-9 / LZMA / zstd-22 / bz2 / PPMd; we report the two strong ones:
   **LZMA** (raw LZMA2, preset 9 | EXTREME) and **zstd-22**.

So **Kalcher = freq-remap → LEB128 → {LZMA, zstd-22}**, vs the repo's
**`+freq` = freq-remap → streamvbyte** (no general compressor).

Latency is agent-facing: decode stops at token IDs (no detokenize), encode
starts from token IDs (no tokenize) — same convention as the repo's other
token-native latency numbers. Ratio excludes any length header (LEB128 is
self-delimiting; LZMA/zstd frames store their own size).

## Results (r50k, 512-token test chunks, median [90% bootstrap CI])

### Compression ratio vs UTF-8

| Method            | English (prose)    | Code               | Hindi              |
| ----------------- | ------------------ | ------------------ | ------------------ |
| raw               | 2.26× [2.20, 2.31] | 1.06× [1.04, 1.13] | 0.85× [0.85, 0.85] |
| +freq             | 2.65× [2.56, 2.70] | 1.40× [1.36, 1.46] | 1.34× [1.34, 1.35] |
| +ANS              | 3.26× [3.19, 3.36] | 2.21× [2.12, 2.30] | 2.80× [2.77, 2.84] |
| **Kalcher(LZMA)** | 3.16× [3.06, 3.22] | 3.01× [2.85, 3.11] | 2.88× [2.83, 2.90] |
| **Kalcher(zstd)** | 3.12× [3.00, 3.16] | 2.90× [2.74, 3.00] | 2.90× [2.82, 2.97] |

`raw`/`+freq`/`+ANS` reproduce the blog numbers (raw 2.27×, +freq 2.62×,
+ANS 3.26× on English — small deltas are just chunk-selection RNG order).

### Decode latency to token IDs (µs/chunk, median-of-30-reps)

| Method            | English  | Code     | Hindi    |
| ----------------- | -------: | -------: | -------: |
| raw               |    0.4   |    0.4   |    0.4   |
| **+freq**         |    4.2   |    4.1   |    4.1   |
| +ANS              |   30.7   |   29.4   |   28.2   |
| **Kalcher(LZMA)** |   72.1   |   57.7   |   52.6   |
| **Kalcher(zstd)** |   35.6   |   31.5   |   31.6   |

Encode latency (µs/chunk): +freq ~3µs, +ANS ~4.6µs, Kalcher(zstd) ~85–120µs,
**Kalcher(LZMA) ~31,000µs** (preset 9|EXTREME compression is inherently heavy;
decode stays cheap, which is what matters at read time).

## Verdict

The data supports the paper's framing: **Kalcher buys ratio with a much slower
decode.** On English, Kalcher(LZMA) reaches 3.16× vs `+freq`'s 2.65×, but its
decode is **17× slower** (72µs vs 4.2µs) and even Kalcher(zstd) at a similar
3.12× decodes ~8× slower (36µs). `+freq` keeps most of the token-native win at
streamvbyte speed.

**Surprise:** on **code** and **Hindi**, Kalcher does not just beat `+freq`, it
**beats `+ANS` on ratio** (code 3.01× vs 2.21×; Hindi 2.90× vs 2.80×). The
general compressor's LZ77 substring matching captures repetition that the
order-0 static ANS/streamvbyte models can't — most pronounced in code, where
the raw token stream is highly repetitive. So Kalcher's ratio advantage is real
and largest exactly where token IDs alone struggle; the cost is a general-purpose
decompressor on every read (30–70µs) instead of a variable-int unpack (~4µs).
Net: `+freq` remains the latency-optimal token-native codec; Kalcher is the
ratio-optimal one for read-cold / write-once archival data where decode speed
matters less.
