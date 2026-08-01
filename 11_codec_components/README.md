# 11 — Byte-codec compress/decompress components (real C4)

Decomposes the **byte-storage path** into its `compress` / `decompress` halves
for every byte codec on **real English C4 chunks**, then composes the full agent
read/write — so we can show honestly that the byte codecs which *match*
token-native's ratio (zstd-19, brotli-q11, zstd --train) are **slow to write**.

Run: `uv run python 11_codec_components/bench_codec_components.py` → `results.json`.

Setup: English C4 `prose_test`, 512-token chunks, 50 chunks, seed 3344, single
core. Codec halves timed **warm / median-of-30** (matches
`bench_unified_latency.py`'s convention for the small deterministic codec ops);
compressed bytes are from real chunks (not synthetic/repetitive text). Full path
composed with the **serving-cold** tokenize/detokenize from `09_cold_tokenize`
(r50k, cache-evicted single shot — the real serving condition):

- **byte write = detokenize + compress**
- **byte read = decompress + tokenize**

## Results — median µs/chunk, real English C4

serving-cold tokenize = **252.3 µs**, detokenize = **40.5 µs** (r50k).

| codec           | compress µs | decompress µs | ratio | compressed B | **full write** (detok+compress) | **full read** (decompress+tok) |
| --------------- | ----------: | ------------: | ----: | -----------: | ------------------------------: | -----------------------------: |
| LZ4             | 5.0         | 1.8           | 1.28× | 1834         | 45.4                            | 254.1                          |
| gzip-9          | 37.0        | 11.2          | 1.95× | 1196         | 77.4                            | 263.6                          |
| zstd-19         | 209.3       | 4.4           | 1.97× | 1184         | 249.8                           | 256.7                          |
| brotli-q11      | **2719.6**  | 9.1           | 2.62× | 900          | **2760.1**                      | 261.4                          |
| zstd --train    | 375.5       | 3.4           | 2.68× | 877          | 416.0                           | 255.7                          |
| **token-native reference** (bench_unified_latency.py) |
| r50k raw        | —           | —             | —     | —            | **0.6**                         | **0.4**                        |
| r50k +freq      | —           | —             | —     | —            | **3.2**                         | **4.0**                        |
| r50k +ANS       | —           | —             | —     | —            | **4.4**                         | **28.6**                       |

## What the decomposition shows

**Read side — decompress is negligible; byte read is tokenize-bound for every
codec.** Decompress is 1.8–11.2 µs across the board (zstd-19 4.4 µs, brotli 9.1 µs,
zstd --train 3.4 µs), so full byte read is **254–264 µs for all five codecs** —
essentially the ~252 µs tokenize, unchanged by codec choice. Accounting for
decompression does **not** move the byte-vs-token read comparison; the gap is the
mandatory tokenize (~250 µs vs token-native's 0.4–28.6 µs), same for any codec.

**Write side — compress is where the ratio-winners pay.** Compress is cheap only
for the low-ratio codecs: LZ4 5 µs (but just 1.28×) and gzip-9 37 µs (1.95×). The
codecs that actually *match token-native's ratio* are expensive to write:
**zstd-19 209 µs, zstd --train 376 µs, and brotli-q11 2,720 µs** to compress a
single 512-token chunk. So full byte write is **250 µs (zstd-19), 416 µs (zstd
--train), 2,760 µs (brotli-q11)** — versus token-native's **0.6–4.4 µs**, a
**60×–4,300×** gap. (Note: on *real* C4, zstd-19 compress is ~209 µs, not the
~11 µs a repetitive synthetic text suggested — real prose is far less
compressible, so the write cost is real.)

## Honest one-sentence takeaway

Accounting for text compress/decompress leaves the **read** comparison
unchanged (decompress is 2–11 µs, so byte read stays ~250 µs tokenize-dominated
for every codec) but decisively changes the **write** comparison: LZ4 is the
latency-charitable byte codec in the table, and any codec you'd actually pick for
good ratio (zstd-19, zstd --train, brotli-q11) makes the byte **write** 60×–4,300×
slower (210–2,720 µs compress) than token-native's 0.6–4.4 µs.
