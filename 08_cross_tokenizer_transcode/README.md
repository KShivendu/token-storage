# 08 — Cross-tokenizer transcoding latency

Measures (and tries to beat) the latency of turning **source-tokenizer IDs into
target-tokenizer IDs** — the "a consumer with a different tokenizer detokenizes
and re-tokenizes" path from the Token-Native Storage post. If a stored payload
is r50k token IDs but the reader speaks o200k, it must transcode.

Run: `uv run python 08_cross_tokenizer_transcode/bench_transcode.py` → `results.json`.

Setup: English (C4) `prose_test`, 512-token chunks, 40 held-out chunks, seed
3344, single core (`RAYON_NUM_THREADS=1`), tiktoken. Latency = median µs/chunk,
each chunk timed as the median of 25 reps, bootstrap 90% CI across chunks.
Directions: **r50k→o200k** (coarse→fine vocab) and **o200k→r50k** (fine→coarse).

## Correctness is the hard constraint

A transcoder is valid only if `transcode(source_ids(T)) == target.encode(T)`
for every text `T`. Every strategy is scored against that ground truth. A
fast-but-wrong transcoder is useless for interchange, so each reports the
fallback rate needed to reach 100%.

## Results (median µs/chunk)

| direction   | baseline | per-token LUT | per-piece memo | decode-only | encode-only |
| ----------- | -------: | ------------: | -------------: | ----------: | ----------: |
| r50k→o200k  | **80.4** | 13.1          | 200.6          | 7.5         | 71.2        |
| o200k→r50k  | **116.6**| 12.6          | 157.9          | 7.3         | 106.2       |

| strategy       | direction  | µs   | speedup | exact-match | fallback to 100% | table/mem |
| -------------- | ---------- | ---: | ------: | ----------: | ---------------: | --------: |
| baseline       | r50k→o200k | 80.4 | 1.00×   | 100%        | —                | 0         |
| baseline       | o200k→r50k | 116.6| 1.00×   | 100%        | —                | 0         |
| per-token LUT  | r50k→o200k | 13.1 | 6.1×    | **0%**      | 100%             | 0.64 MB   |
| per-token LUT  | o200k→r50k | 12.6 | 9.2×    | **2.5%**    | 97.5%            | 4.2 MB    |
| per-piece memo | r50k→o200k | 200.6| **0.40×** | **100%**  | 0% (correct)     | 15.0 MB   |
| per-piece memo | o200k→r50k | 157.9| **0.74×** | **100%**  | 0% (correct)     | 14.5 MB   |

## Strategy notes

**Baseline — `target.encode(source.decode(ids))`.** Lossless by construction,
100% exact. Decomposes into a cheap Rust `decode` (~7µs) and a Rust `encode`
(71µs r50k→o200k, 106µs o200k→r50k). The encode dominates, and it is already
native-optimized. **This is the number to beat.**

**Per-token LUT (concatenate a precomputed per-source-token target-ID table).**
Fastest in raw terms (6–9× the baseline: 13µs) because it's pure dict lookups +
list concat, no tokenization. But it is **wrong**: chunk-level exact-match is
**0% (r50k→o200k)** and **2.5% (o200k→r50k)**; even token-level agreement is
only 5–18%. BPE merges are context-dependent and never cross regex-piece
boundaries, so tokenizing each source token in isolation and concatenating the
IDs diverges at almost every word-internal boundary (and o200k's partial-byte
tokens don't even decode in isolation). To make it correct you'd have to verify
each chunk against the real `target.encode`, i.e. pay the baseline anyway — the
"correct-with-verify" latency is **93µs / 129µs, *slower* than baseline
(0.86× / 0.90×)**. Not viable.

**Per-piece memoization (the sound unit).** tiktoken encodes by regex-splitting
into word-ish "pieces" and running BPE per piece; merges never cross piece
boundaries, so concatenating per-piece target IDs **equals** the full encode
(validated: 100% exact both directions). A `{piece_text → target_ids}` cache
warmed on the full train split gives a **100% piece hit-rate** on the test set,
so it skips BPE entirely at runtime — yet it is **2.5× slower (r50k→o200k) and
1.35× slower (o200k→r50k)** than baseline. The reason: the Python `regex.findall`
split + ~300 dict lookups + list-building per chunk costs more than tiktoken's
single native `encode` call. It proves the piece is the correct memoization
unit and achieves 100% correctness, but Python overhead sinks the latency.
Cache cost: ~200K entries, ~15 MB.

## Verdict

**Full retokenize is basically the floor: cheap lossless cross-tokenizer
transcoding is not feasible here — the only thing faster than baseline (per-token
LUT, 6–9×) is hopelessly wrong (0–2.5% exact) and turns *slower* than baseline
once you add the verification needed to fix it, while the one correct shortcut
(per-piece memoization, 100% exact, 100% cache hits) still loses to baseline
because tiktoken's BPE encode is already native-fast and any Python-level
per-piece logic costs more than the Rust call it replaces.** A genuinely faster
lossless transcoder would need to be native (operate on the BPE merge state
directly) and, even then, BPE's context-dependent merges mean you must still
re-run BPE within each regex piece — so the theoretical ceiling is roughly the
regex-split cost you could save, a small slice of an already-cheap operation.
