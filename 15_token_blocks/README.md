# 15 — Token-native methods over ES/Lucene-style 16 KB blocks

`07` measures 16 KB blocks for **byte codecs only**, and the 2x2 read table has
a single `token/block` latency with no ratio. So every ratio comparison in the
talk pitted per-document token storage against *blocked* byte storage, which is
not the fight a real search engine presents. This fills that in.

Documents are grouped exactly as Lucene groups them, and exactly as `07` does:
corpus-adjacent order, flush at 16 KB of raw UTF-8 or 128 docs, trailing
partial block still compressed. **Both stores hold the same documents per
block**, so read amplification is identical and the ratios are comparable. The
only thing that differs is what gets written — the block's UTF-8 bytes, or the
block's concatenated token IDs.

Run: `uv run python 15_token_blocks/bench_token_blocks.py` → `results.json`

Models (rank table, ANS model, +dict dictionary) are trained on the `_train`
split only. Note `+dict` here is a 112 KB zstd dictionary trained on **packed
token-ID bytes**, matching `03_latency`'s definition — not `tnbench`'s
`full_train_zstd_dict`, which trains on UTF-8 and is the byte-side
`zstd --train`.

## Block ratios (raw UTF-8 / compressed block)

| method | prose (6 docs/blk) | code (14) | hindi (18) |
|---|---:|---:|---:|
| byte LZ4 | 1.45 | 2.01 | 1.92 |
| byte gzip-9 | 2.17 | 2.99 | 3.70 |
| byte zstd-19 | 2.25 | 3.06 | 3.92 |
| tok raw | 2.28 | 1.37 | 2.55 |
| tok `+freq` | 2.66 | 2.29 | 4.46 |
| tok `+ANS` | **3.29** | 2.71 | 4.92 |
| tok `+freq`→LZ4 | 2.68 | 2.87 | 4.49 |
| tok `+dict` | 3.20 | **3.59** | **4.93** |

Byte LZ4 at 1.45x on prose reproduces `07`'s 1.43x, so the two block
simulations agree.

## The mechanism: blocking helps LZ, not order-0

Per-document (512-token) against the same method blocked:

| | per-doc | blocked | change |
|---|---:|---:|---|
| prose zstd-19 | 1.95 | 2.25 | **+15%** |
| prose `+ANS` | 3.38 | 3.29 | −3% |
| code zstd-19 | 2.48 | 3.06 | **+23%** |
| code `+ANS` | 3.05 | 2.71 | −11% |
| code `+dict` | 3.39 | 3.59 | +6% |

A bigger block is a bigger LZ77 window, so anything with an LZ layer gains.
A static order-0 entropy coder has no window to widen, so `+ANS` gains nothing
— it is already coding each token at its unigram entropy. `+dict` gains because
zstd's LZ layer finds cross-document repeats in the *ID* stream.

That is also why `+dict` is the one to quote on code: order-0 loses there, and
blocking is exactly the condition that rewards modelling repetition.

## Verdict, blocked against blocked

- **prose**: `+ANS` 3.29x vs best byte 2.25x — token wins by **46%**
- **code**: `+ANS` 2.71x *loses* to zstd-19's 3.06x; `+dict` 3.59x wins by **17%**
- **hindi**: `+dict` 4.93x vs 3.92x — token wins by **26%**

So the thesis survives the fair fight, but on code it survives only via
`+dict`. Do not claim `+ANS` beats a blocked byte store on code.

Encode is not close: on code, `+ANS` costs 2.4 µs/doc and `+dict` 140.6 µs/doc
against zstd-19's 266.1 µs/doc.

## Not measured here

Decode. For the block-level read comparison see
`07_kalcher_baseline/read_latency_2x2_results.json`: `token/block` 39.8 µs
against `byte/block` 304.8 µs on prose, the gap being the tokenize the byte
path still owes.
