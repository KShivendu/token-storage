# 14 — Hybrid: frequency remapping + LZ4

`+freq` is order-0. It reassigns token IDs by frequency rank and packs them,
modelling no repetition at all, which is why it falls behind on code where the
same imports and signatures recur constantly. `+lz4` is the opposite: it finds
repeats but ignores the frequency skew. Can they be stacked?

**Yes, and on code it is close to free.**

Run: `uv run python 14_freq_lz4_hybrid/bench_freq_lz4_hybrid.py` → `results.json`

Setup: 512-token chunks in each corpus's native tokenizer, 60 chunks, rank
table built on `*_train` and evaluated on `*_test` (so the frequency table
never sees its evaluation data). Every variant is **round-trip asserted** back
to the exact token IDs. Warm median-of-30. Expect a few % run to run.

## Results (ratio vs raw UTF-8, µs per 512-token chunk)

| variant | prose | code | hindi | encode | decode |
|---|---:|---:|---:|---:|---:|
| raw (fixed pack) | 2.25 | 1.44 | 2.26 | 7.5 | 5.8 |
| `+lz4` (raw → LZ4) | 2.37 | 2.46 | 2.71 | 10.2 | 7.2 |
| `+freq` (svb) | 2.65 | 2.41 | 3.91 | 3.5 | 4.4 |
| `+freq` fixed → LZ4 | 2.38 | 2.50 | 2.85 | 11.1 | 8.0 |
| **`+freq` svb → LZ4** | 2.64 | **2.98** | **4.04** | 5.3 | 5.4 |
| **`+freq` leb128 → LZ4** | **2.90** | **3.36** | **4.35** | 40.4 | 32.4 |
| `+freq` fixed → LZ4-HC | 2.40 | 2.65 | 2.95 | 37.0 | 8.0 |

*(latency columns are the code corpus)*

## The practical winner: `+freq` → streamvbyte → LZ4

On code it lifts **2.41x → 2.98x (+24%)** for **+1.8 µs encode and +1.0 µs
decode**. Hindi gains a little (3.91 → 4.04), prose is a wash (2.65 → 2.64).
Given the decode budget this competes against is a 227 µs tokenize, a 1 µs
decode increase for a quarter more compression on the corpus we were losing is
close to free.

## The ratio winner: `+freq` → LEB128 → LZ4

Best ratio everywhere — code **3.36x**, which essentially matches `+dict`'s
3.39x **without needing a trained dictionary shipped alongside the data**, and
at 2.5x cheaper encode (40 µs vs 102 µs). It costs decode: 32 µs against
`+dict`'s 6.3 µs.

The 40 µs is not a slow-language artifact — `leb128_encode` is fully vectorized
numpy. It is numpy call overhead on a 512-element array, against a *compiled*
streamvbyte. A compiled LEB128 should land far closer to svb, which would make
this the outright best option. Worth building before drawing conclusions.

## What did not work, and why it is interesting

**Fixed-width remapped IDs → LZ4 was among the worst** (code 2.50x), despite
the appealing theory: keeping byte alignment should help LZ4 find matches, and
with uint16 any rank under 256 leaves a zero high byte for LZ4 to eat.

The theory loses to arithmetic. Padding every token to a fixed width inflates
the stream more than the extra matches recover. The compact varint encodings
win even though they *destroy* byte alignment and shorten matches below LZ4's
4-byte minimum. Compactness beats alignment here.

LZ4-HC on the same fixed stream does not rescue it either (2.65x on code, at
37 µs encode) — confirming the problem is the representation, not the search.

## Caveat

These beat `+freq` but do not beat `+dict` on code (3.39x), and none of them
touch `zstd --train`'s 3.34x→4.78x at large chunks. The appeal is that they
need only the rank table you already have, no dictionary to train, version and
ship with the corpus.
