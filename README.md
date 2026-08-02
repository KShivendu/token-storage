# token-storage

Experiments behind [Token-Native Storage](https://www.kshivendu.dev/blog/token-storage): does storing text as BPE token IDs instead of UTF-8 bytes get you free compression and lower agent read/write latency?

## Method

- **Corpora**: English ([C4](https://huggingface.co/datasets/allenai/c4)), code ([codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean)), Hindi ([Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)) — held-out train/test splits, nothing trains on test.
- **Tokenizers**: r50k, cl100k, o200k (`tiktoken`), Qwen2.5, DeepSeek-V2, Gemma-2, BERT-WordPiece (HF configs).
- **Chunk sizes**: 512 tokens (main results), also swept at 256/2000.
- 40 sampled test chunks/domain, bootstrap CI (2000 resamples, 90%), median-of-30-reps for sub-µs latencies.
- Everything lossless except WordPiece (see findings).

**Canonical numbers.** Ratios come from `07_kalcher_baseline/results.json` →
`table1_full_train_consistent`, one harness at seed 9012 with every cell in the
same setup. Latencies come from `03_latency/latency_grid_results.json`. Older
files under `01_storage_efficiency/` retain earlier partial-train runs and will
differ by a few percent — prefer the two above.

## Findings

| Question | Result |
| --- | --- |
| Compression vs. raw UTF-8 (English, 512-tok, median) | LZ4 1.27x, gzip 1.92x, zstd 1.94x, brotli 2.57x, zstd --train 2.72x, r50k raw **2.25x**, r50k +ANS **3.30x** |
| Does it hold across tokenizers/vendors? | Yes — r50k/cl100k/o200k/Qwen2.5/DeepSeek-V2/Gemma-2 all land **3.30–3.40x** with static ANS. The per-cell 90% CIs are ±0.1, so the six are statistically indistinguishable. |
| Where does the gain come from, coder or tokenizer? | Tokenizer: ANS on raw bytes only reaches 1.76x; feeding it tokens instead of bytes gets to 3.30x |
| Cheaper alternative to ANS (frequency-remap + streamvbyte) | **~21% less ratio** (English: 2.60x vs 3.30x), **~7x faster decode** (3.4µs vs 23.3µs) |
| Does the tokenizer have to cover the script? | Yes for raw packing — r50k on Hindi is 0.84x, *worse* than UTF-8. o200k gets 2.55x raw / 5.90x +ANS on the same text. Either coder repairs a mismatch (r50k Hindi: 1.33x +freq, 2.97x +ANS). |
| Best on code? | Not the order-0 coders. Code is repetitive, so `zstd --train` (3.47x) and Kalcher+LZMA (3.44–3.46x) beat +ANS (3.19x). A token-domain dictionary closes it: `+dict` 3.55x, `dict-freqvarint` 3.90x. These overlap within CI — no clear winner on code. |
| Write latency, agent as writer | Token-native **0.4–5.9µs** (the writer already produced the IDs) vs ~49µs for an LZ4 byte store, which must detokenize (46µs) before compressing |
| Read latency, agent as reader | Token-native **0.3–32µs**. A byte store pays **~242µs/read** (r50k), essentially all of it re-tokenizing text the model will immediately re-encode. |
| Do embedding-model tokenizers (WordPiece) compress too? | Yes, slightly better ratio (3.56x), but **80.5% char-error-rate** on decode (BERT lowercasing) — not lossless, unusable for storage |

Full breakdowns, all three domains, all three chunk sizes: see the numbered
directories, `00_corpus_prep` through `10_code_dict`.

## Latency: always run pinned

These are single-core benchmarks. On a hybrid CPU (this box is a Core Ultra 7
155H: P-cores 4.8 GHz, LP-E cores 2.5 GHz) an unpinned run can be migrated onto
an LP-E core and every cell inflates ~1.6x. Pin to a top-frequency core:

```bash
taskset -c 4 env RAYON_NUM_THREADS=1 TIKTOKEN_MAX_THREADS=1 \
  uv run python 03_latency/bench_latency_grid.py
```

**Tokenize is measured serving-cold**, via `tnbench.timed_once_serving`: a 64 MB
cache sweep before each single shot, so the tokenizer's multi-MB rank table is
evicted as it is in real serving, where each read is interleaved with codec and
model work. This is the condition the paper reports. A back-to-back tokenize loop
instead keeps that table resident and reports roughly **half** the real cost, so
do not use plain `timed_once` for tokenize. See `09_cold_tokenize/` for the full
text-cold vs serving-cold characterisation.

Serving-cold English, 512-token chunk, per tokenizer. Two independent harnesses,
same protocol, P-core pinned:

| | r50k | cl100k | o200k |
| --- | --: | --: | --: |
| tokenize (`03_latency` grid) | 240 µs | 306 µs | 281 µs |
| tokenize (`09_cold_tokenize`) | 248 µs | 291 µs | 267 µs |
| detokenize (`03_latency` grid) | 46 µs | 51 µs | 56 µs |
| detokenize (`09_cold_tokenize`) | 45 µs | 44 µs | 51 µs |

`07_kalcher_baseline`'s interleaved loop gets 283 µs for r50k without a synthetic
evictor, so **~240–283 µs** is the honest r50k band. Quote a band, not a point.

## Setup

```bash
uv sync
uv run python 00_corpus_prep/prep_multidomain.py  # builds data/corpus/, run once
```

Each numbered directory is runnable independently after that and maps to one section of the post.

## CLI

`bench.py` at the repo root is a thin config + dispatch layer over the existing
benches (it imports and calls `run_cell` / `byte_codec_components` and the
block/2x2 flags, and defines no measurement logic of its own). Tokenizer
selection is always a list mapped per corpus, so a cross-corpus run can never
fall back to a single hardcoded tokenizer.

```bash
# compression ratio (Table 1), all tokenizers x all corpora, 512-token chunks
uv run python bench.py --experiment ratio

# agent read/write latency (Table 2), pinned, r50k only, English only
taskset -c 4 env RAYON_NUM_THREADS=1 TIKTOKEN_MAX_THREADS=1 \
  uv run python bench.py --experiment latency --tokenizers r50k --corpora prose

# ratio + latency across chunk sizes (defaults to 512 1024 2048 4096)
taskset -c 4 uv run python bench.py --experiment chunk-sweep

# byte/token x document/block 2x2 (ratio + single-document read latency)
uv run python bench.py --experiment block2x2

# pick a subset of methods, sizes, seed
uv run python bench.py --experiment ratio --methods LZ4 zstd-19 +ANS +dict --chunk-sizes 512 2048
```

Knobs: `--experiment {ratio,latency,block2x2,chunk-sweep}`, `--tokenizers`,
`--corpora`, `--chunk-sizes`, `--methods`, `--seed`, `--n-chunks`, `--out`. The
script prints its resolved config before running. See `--help` for details.

## License

MIT. Cite the blog post if you use these numbers elsewhere.
