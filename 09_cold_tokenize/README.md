# 09 — Cold tokenize / detokenize latency across tokenizers

The serving-latency argument in the Token-Native Storage post charges the byte
path a **cold** tokenize: each retrieved chunk is a fresh, distinct text the
tokenizer sees exactly once. This benchmark measures that cold cost — **both
directions, tokenize (encode) and detokenize (decode)** — for **r50k, cl100k,
o200k**, so the paper can state the cost across tokenizers honestly.

Run: `uv run python 09_cold_tokenize/bench_cold_tokenize.py` → `results.json`.

Setup: English (C4) `prose_test`, **512-token chunks in each tokenizer's own
encoding** (a stored chunk is 512 IDs of that tokenizer), 100 distinct chunks,
seed 3344, single core (`RAYON_NUM_THREADS=1`). Median µs/chunk, bootstrap 90% CI.

## Headline

> Tokenizing a fresh 512-token English chunk costs **~120 µs** (r50k), detokenizing
> **~20 µs**, single core — when the tokenizer's ranks table is warm in CPU cache.
> Under the real serving mix, where each read interleaves decompression and other
> work that evicts that table, cold tokenize rises to **~210 µs** and detokenize to
> **~40 µs** (the paper's ~283 µs r50k figure is this cache-evicted condition).

## Two cold definitions (they differ ~2×, and the difference matters)

The methodology follows `07_kalcher_baseline/bench_unified_latency.py`'s
`timed_once`: **one reused `Encoding` object** (tokenizer built, ranks table +
regex loaded in RAM) and each chunk's `encode`/`decode` timed with a **single
`perf_counter` shot** so the *text* is cold on first touch. But a back-to-back
tokenize loop also keeps the ranks table **hot in CPU cache**, which real
serving does not:

- **Text-cold** (this loop): text unseen, but ranks table CPU-cache-warm.
- **Serving-cold**: CPU cache evicted between reads (64 MB sweep), so the
  multi-MB ranks table is cold too — reproducing the paper's mixed-loop
  condition where tokenize is interleaved with decompress/codec work.

## Results (median µs/chunk, English C4, 512-tok, single core)

### Tokenize (encode, IDs ← text)

| tokenizer | text-cold | serving-cold | warm (30-rep) | serving-cold / warm | cold µs / 1k UTF-8 B | median UTF-8 B/chunk |
| --------- | --------: | -----------: | ------------: | ------------------: | -------------------: | -------------------: |
| r50k      | 116.1     | **207.9**    | 106.2         | 1.96×               | 49.6                 | 2344 |
| cl100k    | 131.4     | **246.4**    | 123.9         | 1.99×               | 54.3                 | 2406 |
| o200k     | 121.9     | **235.2**    | 81.5          | 2.87×               | 50.7                 | 2438 |

### Detokenize (decode, text ← IDs)

| tokenizer | text-cold | serving-cold | warm (30-rep) | cold / warm |
| --------- | --------: | -----------: | ------------: | ----------: |
| r50k      | 20.8      | 42.2         | 7.6           | 2.72× |
| cl100k    | 30.5      | 45.1         | 7.7           | 3.97× |
| o200k     | 20.8      | 51.3         | 7.9           | 2.62× |

(µs-per-1k-UTF-8-byte is reported because o200k/cl100k pack slightly more text
into 512 tokens; per byte all three land at **~50 µs/1k UTF-8 B** to tokenize
cold — the per-chunk and per-byte views tell complementary stories.)

## Reading the numbers

- **Tokenize dominates; detokenize is cheap.** Cold tokenize is ~120–250 µs;
  cold detokenize is ~20–50 µs (2–6× cheaper). Both are mandatory on the byte
  path for a model reader, and both are *free* on the token-native path (the
  stored IDs are already what the model consumes).
- **The cold/warm gap is a CPU-cache effect, not a warm-up-the-object effect.**
  Text-cold-but-cache-warm tokenize is only ~1.1–1.5× the fully warm number;
  the big ~2–2.9× cold/warm ratio the post cites is **serving-cold** — the ranks
  table being evicted from cache by the decompression/other work between reads.
  For a serving-latency argument (reads interleaved with real work), the
  **serving-cold column (~210–250 µs tokenize) is the honest number**; the
  text-cold column is the floor you'd only hit in a tight tokenize loop.
- **All three tokenizers are within ~1.2× of each other** on cold tokenize
  (r50k 208, o200k 235, cl100k 246 µs serving-cold), and essentially identical
  (~50 µs/1k B) once normalized per byte — the "byte path pays a few-hundred-µs
  cold tokenize per retrieved chunk" claim holds across vocabularies, not just
  r50k.
- **Caveat:** single-shot cold timings are inherently noisy; 100 distinct chunks
  + bootstrap smooth it, but expect ±10–20 µs run-to-run on the cold columns.
  The serving-cold proxy (64 MB sweep) is a synthetic cache-evictor; the paper's
  283 µs came from real interleaved codec work, so treat ~210–283 µs as the
  serving-cold band for r50k rather than a single point.
