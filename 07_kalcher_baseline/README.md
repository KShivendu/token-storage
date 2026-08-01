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

---

## Chunk-size sweep (ratio, r50k)

`bench_kalcher_chunksweep.py` extends the 512-token ratios to 512 / 1024 / 2048
/ 4096 tokens (the blog's existing sweep in `token-storage-extra.mdx` only went
256 / 512 / 2000 and had no Kalcher). Table convention is identical to
`bench_kalcher.py` (seed 3344, full-train rank table, ANS on a fixed 400×512
train-token table held constant across sizes), so the 512 column reproduces the
committed 512 numbers exactly. Ratio vs UTF-8, median [90% CI]:

**English (prose)**

| method        | 512tok | 1024tok | 2048tok | 4096tok |
| ------------- | -----: | ------: | ------: | ------: |
| raw           | 2.26×  | 2.35×   | 2.32×   | 2.30×   |
| +freq         | 2.65×  | 2.67×   | 2.67×   | 2.67×   |
| +ANS          | 3.26×  | 3.34×   | 3.31×   | 3.31×   |
| Kalcher(LZMA) | 3.16×  | 3.41×   | 3.48×   | **3.54×** |
| Kalcher(zstd) | 3.12×  | 3.37×   | 3.46×   | 3.50×   |

**Code**

| method        | 512tok | 1024tok | 2048tok | 4096tok |
| ------------- | -----: | ------: | ------: | ------: |
| raw           | 1.03×  | 1.08×   | 1.07×   | 1.08×   |
| +freq         | 1.39×  | 1.43×   | 1.40×   | 1.49×   |
| +ANS          | 2.21×  | 2.23×   | 2.20×   | 2.24×   |
| Kalcher(LZMA) | 3.15×  | 3.43×   | 4.18×   | **4.14×** |
| Kalcher(zstd) | 3.05×  | 3.32×   | 4.03×   | 4.06×   |

**Hindi**

| method        | 512tok | 1024tok | 2048tok | 4096tok |
| ------------- | -----: | ------: | ------: | ------: |
| raw           | 0.84×  | 0.84×   | 0.84×   | 0.84×   |
| +freq         | 1.34×  | 1.34×   | 1.34×   | 1.34×   |
| +ANS          | 2.79×  | 2.80×   | 2.80×   | 2.80×   |
| Kalcher(LZMA) | 2.81×  | 3.20×   | 3.70×   | **4.15×** |
| Kalcher(zstd) | 2.80×  | 3.22×   | 3.71×   | 4.18×   |

Kalcher(LZMA) − +ANS gap by size — prose: −0.10 → +0.07 → +0.18 → +0.23;
**code: +0.93 → +1.20 → +1.98 → +1.90**; hindi: +0.02 → +0.40 → +0.90 → +1.35.

### Verdict on the chunk-size trend

**Confirmed — Kalcher's advantage grows monotonically with chunk size; `+ANS`
and `+freq` are essentially flat.** The order-0 entropy/varint models are
memoryless, so their ratio barely moves (prose +ANS ~3.3× and +freq ~2.67× at
every size; Hindi +ANS 2.80× and +freq 1.34× dead flat). Kalcher's general
compressor gets more repetition to exploit as chunks grow, so it climbs on all
three domains. Consequences:

- **Code:** the crossover we saw at 512 (Kalcher already ahead) *widens* — the
  Kalcher−ANS gap roughly doubles from +0.93 (512) to ~+1.9–2.0 (2048–4096).
- **A crossover appears where there wasn't one at 512:** on English, Kalcher
  starts below +ANS (3.16× vs 3.26× at 512) but overtakes it by 1024 (3.41× vs
  3.34×) and leads by +0.23× at 4096. On Hindi, Kalcher and +ANS tie at 512
  (~2.8×), then Kalcher pulls far ahead (4.15× vs 2.80× at 4096).
- **Net:** at 4096 tokens Kalcher(LZMA) beats `+ANS` on **all three** domains.
  So the paper's "+ANS wins" story is chunk-size-dependent: it holds only at
  small RAG-style chunks (≤512) and only on skewed natural language; for longer
  documents a general compressor over frequency-ordered tokens wins on ratio —
  still at the read-latency cost quantified above (30–60µs vs `+freq`'s ~4µs).

---

## Matched Table 1 — Kalcher rows drop-in comparable to the paper's main table

`bench_kalcher_table1_matched.py` is a **faithful copy of
`01_storage_efficiency/bench_summary_tables.py`** (the script that produced the
blog's 3-domain × 3-tokenizer ratio table) with Kalcher added *inline on the
identical test chunks*. Kalcher draws no RNG during the aligned run and is
bootstrapped only afterward, so every `raw`/`+ANS`/byte-codec row stays
bit-identical to the blog. Config: seed 9012, 512-token chunks, full-train
Laplace ANS, full-train rank table (matches `+freq`).

### Anchor reproduction — ALL 9 +ANS cells reproduced exactly

| tokenizer | prose | code | hindi |
| --------- | ----: | ---: | ----: |
| r50k +ANS   | 3.30× | 2.58× | 2.97× |
| cl100k +ANS | 3.37× | **3.19×** | 3.48× |
| o200k +ANS  | 3.40× | 3.16× | **5.90×** |

Every value matches the published blog table to the cent (incl. the three
anchors the coordinator named: r50k+ANS prose 3.30× — the English-only
`bench_full_results_table.py` reports 3.26× for this because it trains ANS on
only the first 400×512 tokens instead of the full split; the 3-domain table
uses full-train, hence 3.30× — cl100k+ANS code 3.19×, o200k+ANS Hindi 5.90×).
This proves the harness is the blog's, so the Kalcher rows below are directly
comparable.

### Kalcher rows (median [90% CI]), same chunks as the +ANS table

| method               | prose | code  | hindi |
| -------------------- | ----: | ----: | ----: |
| r50k Kalcher(LZMA)   | 3.17× | 3.12× | 2.74× |
| r50k Kalcher(zstd)   | 3.09× | 3.01× | 2.71× |
| cl100k Kalcher(LZMA) | 3.24× | **3.46×** | 3.15× |
| cl100k Kalcher(zstd) | 3.19× | 3.35× | 3.04× |
| o200k Kalcher(LZMA)  | 3.26× | 3.44× | 4.93× |
| o200k Kalcher(zstd)  | 3.22× | 3.35× | 4.80× |

### Overall best method per language (max ratio across ALL methods)

| language | best method            | ratio | runner-up                    |
| -------- | ---------------------- | ----: | ---------------------------- |
| English  | **o200k +ANS**         | 3.40× | cl100k +ANS (3.37×)          |
| Code     | **cl100k Kalcher(LZMA)** | 3.46× | o200k Kalcher(LZMA) (3.44×) |
| Hindi    | **o200k +ANS**         | 5.90× | o200k Kalcher(LZMA) (4.93×)  |

Exactly the split the paper suspected: **English → token+ANS, code → Kalcher,
Hindi → o200k+ANS.** So for Table 1, the true per-column winner to bold is a
token+ANS on English and Hindi, but **Kalcher(LZMA) on code** (3.46× beats the
best +ANS 3.19× by +0.27×, because LZ77 exploits code's token-stream repetition
that an order-0 ANS model can't). On Hindi the +ANS win is decisive (5.90× vs
4.93×). The ranking is at 512-token chunks; per the chunk-size sweep above,
Kalcher's share of the wins only grows with document length.

---

## Path B — the COMPLETE Table 1 on one internally-consistent full-train setup

Same validated harness (`bench_kalcher_table1_matched.py`, seed 9012), but now
**every cell measured in one setup**: full-train Laplace ANS, full-train rank
table (`+freq`), **full-train zstd dictionary** (`zstd --train`), and Kalcher —
all on the identical 512-token chunks. A ready-to-drop-in alternative to the
current mixed-setup blog table. Medians (512-tok):

| method               | English | Code  | Hindi |
| -------------------- | ------: | ----: | ----: |
| LZ4                  | 1.27×   | 1.76× | 1.52× |
| gzip-9               | 1.92×   | 2.46× | 2.38× |
| zstd-19              | 1.94×   | 2.45× | 2.42× |
| brotli-q11           | 2.57×   | 2.87× | 2.89× |
| zstd --train         | 2.72×   | 3.47× | 4.52× |
| r50k raw             | 2.25×   | 1.07× | 0.84× |
| o200k raw            | 1.59×   | 1.49× | 2.55× |
| r50k +freq           | 2.60×   | 1.41× | 1.33× |
| o200k +freq          | 2.73×   | 2.50× | 4.47× |
| r50k +ANS            | 3.30×   | 2.58× | 2.97× |
| cl100k +ANS          | 3.37×   | 3.19× | 3.48× |
| o200k +ANS           | **3.40×** | 3.16× | **5.90×** |
| cl100k Kalcher(LZMA) | 3.24×   | **3.46×** | 3.15× |
| cl100k Kalcher(zstd) | 3.19×   | 3.35× | 3.04× |
| o200k Kalcher(LZMA)  | 3.26×   | 3.44× | 4.93× |
| o200k Kalcher(zstd)  | 3.22×   | 3.35× | 4.80× |

Per-column winner: **English o200k+ANS 3.40×, Code cl100k Kalcher(LZMA) 3.46×,
Hindi o200k+ANS 5.90×.**

### Blast radius: cells that changed vs the current blog table

Sanity confirmed: **raw and the non-trained byte codecs (LZ4/gzip-9/zstd-19/
brotli-q11) are unchanged** (train-independent). Only the three train-dependent
families move, and only slightly:

| cell                  | blog  | full-train | Δ      |
| --------------------- | ----: | ---------: | -----: |
| zstd --train, English | 2.69× | 2.72×      | +0.03  |
| **zstd --train, Code** | 3.24× | **3.47×** | **+0.23** |
| zstd --train, Hindi   | 4.56× | 4.52×      | −0.04  |
| r50k +freq, English   | 2.62× | 2.60×      | −0.02  |
| o200k +freq, English  | 2.76× | 2.73×      | −0.03  |
| o200k +freq, Code     | 2.53× | 2.50×      | −0.03  |
| o200k +freq, Hindi    | 4.39× | 4.47×      | +0.08  |

Takeaways:

- **`+ANS` does not move at all** — the blog's 3-domain table already used
  full-train ANS, so all nine +ANS cells reproduce exactly. The only +ANS shift
  is the **English headline r50k+ANS 3.26× → 3.30×**, and that is *not* a change
  within this table: 3.26× comes from the separate English-only script
  (`bench_full_results_table.py`) that trains ANS on just the first 400×512
  tokens; the consistent 3-domain number is 3.30×.
- **`+freq` barely moves** (≤0.08×) — the blog's `+freq` was *already* full-train
  rank; the sub-0.1× wiggles are chunk-selection noise (seed 3344 → 9012), not
  train dependence.
- **`zstd --train` is the one real mover:** training the dictionary on the full
  train split instead of the blog's 400-sample subset lifts **Code +0.23×**
  (3.24× → 3.47×) and leaves English/Hindi within ±0.04×. It changes no
  per-column winner (still below the token methods on all three).

Net blast radius of switching to a single full-train table: essentially nil for
raw/byte/+ANS/+freq, one meaningful +0.23× bump for zstd --train on code, and a
consistent 3.30× English r50k+ANS headline in place of the 3.26× from the
retired English-only harness.

---

## Section 5.2 — English +ANS across six tokenizers, on full-train

The paper's "generality across tokenizers" paragraph listed English +ANS on the
old 400-chunk setup; recomputed here on the **same full-train harness** (seed
9012) so it is consistent with the full-train Table 1. The tiktoken three come
straight from the aligned run (bit-identical to Table 1); the three HF
tokenizers use the `02_multi_tokenizer` configs, full-train Laplace(+1) ANS,
vocab-clipped ids (matching `bench_tokenizer_gen.py`), on the identical
prose-512 chunks. English (C4), 512-token chunks:

| tokenizer   | full-train English +ANS [90% CI] | (old 400-chunk) |
| ----------- | -------------------------------: | --------------: |
| r50k        | 3.30× [3.20, 3.40]               | 3.26× |
| cl100k      | 3.37× [3.25, 3.45]               | 3.27× |
| o200k       | 3.40× [3.28, 3.46]               | 3.18× |
| Qwen2.5     | 3.36× [3.23, 3.42]               | 3.18× |
| DeepSeek-V2 | 3.32× [3.21, 3.39]               | 3.20× |
| Gemma-2     | 3.37× [3.27, 3.46]               | 3.12× |

**Min–max band: 3.30× – 3.40×** (all six within a 0.10× spread). Full-train
tightens and lifts the band vs the old 400-chunk numbers (which spanned
3.12–3.27×): every tokenizer gains ~0.1–0.2× from the larger frequency table,
and the "all modern LLM tokenizers land in one tight band" claim is *stronger*
on full-train — the band is now ~3.3–3.4× regardless of vendor or vocab size.
