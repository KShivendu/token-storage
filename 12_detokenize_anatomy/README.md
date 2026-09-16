# 12 — Why detokenize costs ~50 µs when it is "just a lookup table"

Detokenizing a 512-token chunk is 512 array lookups and one concat. That should
be a couple of microseconds, yet 09 measures **~45-51 µs** serving-cold. This
benchmark shows where the time goes: **it is not compute, it is DRAM latency**,
and the reason is precisely that it is a lookup table.

Run: `uv run python 12_detokenize_anatomy/bench_detokenize_anatomy.py` → `results.json`.

Setup: English (C4) `prose_test`, 512-token chunks in each tokenizer's own
encoding, 100 chunks, seed 5566. Serving-cold = 64 MB cache sweep, then one
shot (`tnbench.timed_once_serving`). Median, bootstrap 90% CI.

## Headline

> Warm, detokenize costs **~16 ns per token** — exactly the "just a lookup
> table" cost you would predict. Serving-cold it costs **~75-93 ns per token**.
> The extra **~60-90 ns** is one main-memory round trip per token, because 512
> token IDs are 512 scattered probes into a 50k-200k entry table that no longer
> fits in cache. Faster code cannot remove a memory stall.

## The measurement trap

Do **not** compare tiktoken against the flat-arena decoder in absolute terms.
tiktoken's decode is compiled Rust; the flat-arena decode here is a Python list
comprehension, so it loses on interpreter overhead alone (212-250 ns/token) and
that says nothing about memory. Only the two *within-implementation* deltas are
controlled, each holding the code fixed and varying only the memory condition.

## Results — per-token nanoseconds

| tokenizer | vocab | arena | tiktoken warm | tiktoken cold | **A: cold−warm** | flat seq | flat random | **B: random−seq** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r50k   |  50,257 | 0.52 MB | 15.6 | 75.5 | **59.9** | 154.9 | 212.3 | **57.4** |
| cl100k | 100,277 | 1.05 MB | 15.8 | 87.8 | **72.1** | 160.8 | 249.9 | **89.1** |
| o200k  | 200,019 | 2.20 MB | 16.0 | 92.7 | **76.6** | 165.5 | 244.0 | **78.5** |

**A** = same Rust code, same inputs; the only difference is whether the ranks
table survived in cache. **B** = same Python code, same arena, same 512
lookups; the only difference is whether the probes are scattered across the
vocabulary or contiguous.

Two independent methods, both landing at **~60-90 ns per token**, both growing
with vocabulary size. That is a DRAM round trip, and a bigger table means less
of it stays cached.

Note the warm column is flat at ~16 ns across a 4× range of vocabulary size:
the *work* does not depend on how big the table is. Only the *cache behaviour*
does. That is the tell.

## Why a faster tokenizer does not fix this

- **Latency-bound, not throughput-bound.** Even an infinitely fast decoder still
  waits on memory. The warm part is already only ~1/5 of the cold cost, so
  optimizing the code optimizes the small share.
- **The access pattern is inherently random.** Token IDs in real text are
  scattered across the vocabulary; there is no locality to exploit.
- **What would actually help is a memory-system fix, not a code fix:** software
  prefetch (all 512 IDs are known up front, so probes for token *i+k* can be
  issued while *i* is still in flight, turning serial stalls into memory-level
  parallelism), and a more compact table — note the arena column: the actual
  string data is only 0.5-2.2 MB, far smaller than a `HashMap<Rank, Vec<u8>>`
  with a separate heap allocation per entry.

## Why this does not weaken the storage argument

Detokenize is what the token path **pays** (IDs → text, only when a human
reads). Tokenize is what it **avoids** (~250-291 µs serving-cold, on every
agent read, see 09). The same cache effect inflates both, and tokenize is the
one on the hot path — so the honest serving-cold framing makes the gap wider,
not narrower.
