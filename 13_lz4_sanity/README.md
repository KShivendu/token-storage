# 13 — Why LZ4 only reaches 1.27x here, when it is quoted at 2-2.5x

The talk says LZ4 gets **1.27x** on English prose. Public LZ4 benchmarks
routinely show **2-2.5x** on text. This checks whether the repo is wrong.

It is not. The 1.27x reproduces exactly, and the gap is two independent
factors, neither of which applies to the condition the talk measures.

Run: `uv run python 13_lz4_sanity/bench_lz4_sanity.py` → `results.json`.

## Headline

> The public figure needs **a large input AND LZ4-HC**. The talk measures
> **2.3 KB independent chunks with LZ4 fast**, which is what a vector DB
> actually stores. At that size LZ4-HC only moves 1.27x → 1.32x, so the
> compression level is not hiding anything: **input size is the binding
> constraint**, and no LZ4 setting escapes it.

## Ratio vs input size, same corpus (English C4, one contiguous stream)

| input | LZ4 fast | LZ4-HC | gzip -9 |
|---|---:|---:|---:|
| 1 KB | 1.19x | 1.23x | 1.82x |
| 2 KB | 1.22x | 1.28x | 1.91x |
| 8 KB | 1.35x | 1.44x | 2.04x |
| 64 KB | 1.56x | 1.89x | 2.34x |
| 256 KB | 1.60x | 2.14x | 2.52x |
| 1 MB | 1.60x | **2.20x** | 2.55x |
| 4 MB | 1.60x | **2.22x** | 2.56x |

LZ4 is LZ77: it compresses by pointing at earlier repeats inside its window.
A 2.3 KB chunk has almost no earlier text to point at. Whole-file benchmarks
(Silesia, enwik8) are megabytes, where the same phrases recur thousands of
times — that is where 2-2.5x comes from.

Note **LZ4 fast plateaus at 1.60x** even with unlimited input. Reaching 2.2x
needs LZ4-HC as well. So "2-2.5x" is specifically *large file + high
compression*, not a property of LZ4 in general.

## The talk's actual condition

200 independent 512-token chunks, median **2,296 bytes**:

| | ratio |
|---|---:|
| LZ4 fast, per chunk | **1.27x** ← the number in the deck |
| LZ4-HC, per chunk | 1.32x |
| same bytes, one 0.5 MB buffer | 1.61x |

Two cross-checks that the pipeline is sound:

- Per-chunk LZ4 (1.27x at 2,296 B) sits exactly on the size sweep between its
  2 KB (1.22x) and 4 KB (1.31x) points.
- Per-chunk gzip-9 measures 1.92x in the main benchmark; the size sweep gives
  1.91x at 2 KB. Independent code paths, same answer.

gzip beats LZ4 by a lot at small sizes because it adds Huffman entropy coding
on top of LZ77 matching, and entropy coding does not need a large window. That
is the same reason tokenizer+ANS holds its ratio at small chunks while the
LZ-family collapses.

## So is LZ4 fast the fair choice?

Yes, and it is the one production runs. Qdrant, Elasticsearch/Lucene and
Postgres use LZ4 in fast mode for stored fields precisely because the point is
speed. Using LZ4-HC would cost ~10x the encode time to gain 4% at this chunk
size. The block-codec experiment (07) covers the other half of what real
engines do — batching documents into 16 KB blocks — and gets 1.43x.
