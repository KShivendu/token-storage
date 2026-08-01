# 10 — Corpus-wide trained dictionary for fast-decode code compression

**Question:** does a corpus-wide trained zstd dictionary let a *fast-decode*
token-native method match/beat **Kalcher-LZMA** on **code** compression, where
`+freq`/`+ANS` lose because they're zeroth-order and can't capture code's high
verbatim cross-file repetition (imports, boilerplate, idioms, repeated
identifiers)?

**Answer: yes, decisively.** A dictionary trained over the whole code train
split, feeding freq-remapped LEB128 varints into zstd, reaches **3.70× at ~23 µs
decode** — beating Kalcher-LZMA (**3.43× at 38 µs**) on ratio *and* decoding
~1.7× faster. It erases Kalcher's only advantage.

Run: `uv run python 10_code_dict/bench_code_dict.py` → `results.json`.

## Pipeline

`chunk → cl100k IDs → freq-remap (full-train rank table) → LEB128 varint
(contiguous per-token bytes, LZ-friendly — NOT streamvbyte's split control/data
layout) → zstd-22 with a corpus-wide trained dictionary`.

- **Dictionary trained corpus-wide, held out honestly:** trained on the LEB128
  varint bytes of every 512-token chunk of the code **train** split (15,000
  pooled samples), evaluated on 40 held-out **test** chunks. The dict never sees
  test data.
- Methodology matches Table 1 path B (`table1_full_train_consistent`): seed
  9012, 512-token chunks, full-train rank/ANS tables, **cl100k** (the strongest
  Kalcher tokenizer on code). Reference rows reproduce path B within CI
  (Kalcher-LZMA 3.43 vs path-B 3.46; +ANS 3.15 vs 3.19).
- **Decode** = to token IDs (zstd-decompress + LEB128-decode + rank un-permute),
  median-of-30 reps, matching `07_kalcher_baseline/bench_unified_latency.py`.

## Code results (cl100k, 512-tok, median [decode µs/chunk])

| method                              | ratio | decode µs | dict |
| ----------------------------------- | ----: | --------: | ---: |
| +freq (streamvbyte)                 | 2.56× | 2.8       | —    |
| +ANS                                | 3.15× | 13.1      | —    |
| **Kalcher (LEB128+LZMA)** *(ref)*   | 3.43× | 38.1      | —    |
| Kalcher (LEB128+zstd, **no dict**)  | 3.30× | 21.6      | —    |
| **dict-zstd 16K**                   | 3.65× | 21.1      | 16 KB  |
| **dict-zstd 64K**                   | 3.66× | 21.9      | 64 KB  |
| **dict-zstd 112K**                  | **3.70×** | 22.8  | 112 KB |
| **dict-zstd 256K**                  | **3.73×** | 22.5  | 256 KB |
| byte zstd --train 112K (+retokenize)| 3.43× | 94.6      | 112 KB |

**The dictionary's isolated contribution** (the controlled delta): same pipeline
with vs without the dict is **3.30× → 3.70× = +0.40× ratio at unchanged ~22 µs
decode**. That is entirely the corpus-wide dictionary capturing code's verbatim
repetition that per-chunk zstd (and the zeroth-order +freq/+ANS) cannot.

**Dict-size sweep:** even a tiny **16 KB** dict (3.65×) already beats Kalcher-LZMA
(3.43×); returns diminish fast (16K→256K is only 3.65→3.73×). The dict is one
shared table for the whole corpus, so its memory is amortized to ~0 per chunk —
112 KB total buys +0.40× on every code chunk forever.

**vs the byte-domain tie:** byte `zstd --train` reaches the same 3.43× as
Kalcher-LZMA but its decode lands at *text*, so reaching token IDs costs a
re-tokenize (94.6 µs total) — 4× slower than dict-zstd and it loses on ratio too.

## Does the dictionary help on language? (control)

Expected: no — natural language lacks code's verbatim repetition. Reality is
more nuanced (ratio vs no-dict Kalcher-zstd, cl100k, dict-112K):

| domain          | no-dict (Kalcher-zstd) | + corpus dict | Δ       |
| --------------- | ---------------------: | ------------: | ------: |
| English (C4)    | 3.28×                  | 3.47×         | +0.19×  |
| **Hindi (Wiki)**| 3.04×                  | **5.19×**     | **+2.15×** |

- **English C4: barely helps (+0.19×)** — heterogeneous web prose has little
  cross-document verbatim overlap, so the dict finds little to prime, exactly as
  hypothesized. (dict-zstd 3.47× only ties `+ANS` 3.48× here.)
- **Hindi Wikipedia: helps enormously (+2.15×) — the surprise.** Wikipedia is
  highly templated (infoboxes, category tags, stock phrases) and cl100k
  shatters Devanagari into many byte-level tokens, producing long *repetitive*
  varint runs the dictionary captures beautifully. So the real axis isn't
  "code vs language" — it's **high verbatim cross-document repetition (code,
  Wikipedia) vs heterogeneous text (C4)**. Caveat: part of Hindi's jump may
  reflect Wikipedia's near-duplicate/templated content, so treat **code as the
  clean headline** and Hindi as a strong-but-corpus-flavored bonus.

## Verdict

**Yes — a corpus-wide trained token dictionary gives a fast-decode method
(3.70× at ~23 µs) that beats Kalcher-LZMA on code ratio (3.43× at 38 µs) while
decoding ~1.7× faster, erasing Kalcher's only advantage; the win generalizes to
any corpus with heavy verbatim cross-document repetition (also Hindi Wikipedia,
+2.15×) and all but vanishes on heterogeneous web prose (English C4, +0.19×).**
