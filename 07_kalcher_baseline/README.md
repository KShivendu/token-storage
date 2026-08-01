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

**Surprise (r50k only):** on **code** and **Hindi**, r50k-Kalcher beats
r50k-`+ANS` on ratio (code 3.01× vs 2.21×; Hindi 2.90× vs 2.80×) — LZ77
substring matching captures token-stream repetition the order-0 ANS model
can't. But this is an r50k artifact: r50k has no Devanagari vocab and tokenizes
code poorly, so its `+ANS` is weak. With the *best* tokenizer per language the
picture changes — see Task A below. The cost is always a general-purpose
decompressor on every read (30–70µs) instead of a variable-int unpack (~4µs).
Net: `+freq` remains the latency-optimal token-native codec; Kalcher is the
ratio-optimal one for read-cold / write-once archival data where decode speed
matters less.

---

## Task A — Kalcher on cl100k / o200k (apples-to-apples ratio)

`bench_kalcher_multitok.py` (seed 9012, full-train discipline matching
`01_storage_efficiency/bench_summary_tables.py`, so the `+ANS` column
reproduces the blog's 3-domain numbers). Ratio vs UTF-8, median [90% CI]:

| domain | tok    | +ANS               | Kalcher(LZMA)      | Kalcher(zstd)      |
| ------ | ------ | ------------------ | ------------------ | ------------------ |
| prose  | r50k   | 3.36× [3.28,3.43]  | 3.22× [3.10,3.33]  | 3.18× [3.07,3.28]  |
| prose  | cl100k | 3.44× [3.35,3.53]  | 3.30× [3.17,3.38]  | 3.25× [3.13,3.34]  |
| prose  | o200k  | 3.46× [3.39,3.53]  | 3.30× [3.20,3.41]  | 3.25× [3.14,3.35]  |
| code   | r50k   | 2.49× [2.36,2.62]  | 3.03× [2.91,3.24]  | 2.91× [2.81,3.09]  |
| code   | cl100k | 3.05× [2.88,3.18]  | 3.35× [3.15,3.55]  | 3.25× [3.01,3.45]  |
| code   | o200k  | 3.01× [2.87,3.15]  | **3.40× [3.15,3.60]** | 3.30× [3.02,3.52] |
| hindi  | r50k   | 2.98× [2.95,3.01]  | 2.73× [2.70,2.77]  | 2.69× [2.67,2.72]  |
| hindi  | cl100k | 3.54× [3.48,3.57]  | 3.17× [3.16,3.21]  | 3.00× [2.98,3.04]  |
| hindi  | o200k  | **6.06× [5.95,6.33]** | 4.95× [4.76,5.20] | 4.81× [4.69,5.10] |

**Best tokenizer per language (each method at its own best tokenizer):**

| language | best +ANS       | best Kalcher(LZMA) | best Kalcher(zstd) |
| -------- | --------------- | ------------------ | ------------------ |
| English  | 3.46× (o200k)   | 3.30× (o200k)      | 3.25× (o200k)      |
| Code     | 3.05× (cl100k)  | **3.40× (o200k)**  | 3.30× (o200k)      |
| Hindi    | **6.06× (o200k)** | 4.95× (o200k)    | 4.81× (o200k)      |

### Verdict on the paper's §5.1 "wins by a wide margin" claim

It needs **softening — it is not uniformly true.** When both methods use their
best tokenizer:

- **English:** `+ANS` wins, but only ~5% (3.46× vs 3.30×). Not a "wide margin."
- **Code:** Kalcher **wins** (3.40× vs 3.05×, +11%). The claim is reversed here —
  code's highly repetitive token stream is exactly what LZ77 exploits and an
  order-0 entropy coder cannot.
- **Hindi:** `+ANS` wins by a genuinely **wide margin** (6.06× vs 4.95×, +22%).
  A well-fit static entropy model over o200k's skewed Devanagari-token
  distribution beats general-compressor-over-varints decisively.

So "tokenizer+ANS wins by a wide margin" holds strongly only on Hindi; it is a
narrow win on English and a **loss** on code. Recommend rewording to something
like "tokenizer+ANS wins on natural language, by a wide margin where the token
distribution is highly skewed (Hindi); on repetitive corpora (code) a general
compressor over frequency-ordered tokens can edge it out." ANS's advantage is
still bought at ~7× cheaper decode (see Task B), which the paper can keep.

---

## Task B — one unified English latency table

`bench_unified_latency.py`, English (C4), 512-token chunks, seed 3344, one
consistent definition: **agent write** = from token IDs (LZ4 = detokenize +
compress); **agent read** = to token IDs (LZ4 = decompress + tokenize;
`+freq`/Kalcher **include** the rank un-permute, timed as one function). Fast
ops use median-of-30-reps; tokenize/detokenize use single-shot (paper
methodology, so LZ4's mandatory tokenize isn't understated by a warm cache).
Median [90% CI], µs/chunk:

| method            | agent write µs          | agent read µs        |
| ----------------- | ----------------------: | -------------------: |
| LZ4               | 59.0 [57.8, 60.6]       | 283.0 [273.4, 287.8] |
| r50k raw          | 0.6 [0.6, 0.6]          | 0.4 [0.4, 0.4]       |
| r50k +freq        | 3.2 [3.1, 3.3]          | 4.0 [3.9, 4.1]       |
| r50k +ANS         | 4.4 [4.3, 4.5]          | 28.6 [27.8, 29.9]    |
| Kalcher(LZMA)     | 25721.0 [25378, 25879]  | 60.5 [59.6, 62.2]    |
| Kalcher(zstd)     | 88.9 [87.2, 90.5]       | 33.4 [32.9, 34.1]    |

Notes:

- This is the single internally-consistent source for regenerating the paper's
  latency table + figure. The `+freq` read here (4.0µs) folds streamvbyte-decode
  and the rank un-permute into one measurement — that is why it differs from the
  paper's Table 2 (~11µs), which summed two separately-timed ops. Every method
  uses the same fused convention now.
- **Read** is the recurring cost: token-native reads are 0.4–60µs vs LZ4's 283µs
  (dominated by the mandatory tokenize). Kalcher read (33–60µs) is 8–15× slower
  than `+freq` (4µs) — the general-decompressor tax on every read.
- **Kalcher(LZMA) write is ~26 ms/chunk** (preset 9|EXTREME). It is not a viable
  write-time codec; Kalcher(zstd) at 89µs is the practical encoder. Decode stays
  cheap for both, which is what recurs.
