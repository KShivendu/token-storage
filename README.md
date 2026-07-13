# token-storage

Real, runnable experiments behind [Token-Native Storage: Read and Write in the
Language Models Already Speak](https://www.kshivendu.dev/blog/token-storage).

Every number quoted in that post comes from one of the scripts below, run
against real tokenizers (`tiktoken`'s r50k/cl100k/o200k, HuggingFace's
Qwen2.5/DeepSeek-V2/Gemma-2/mxbai), the real `constriction` ANS library, and
real corpora (WikiText-103). Nothing here is estimated or simulated. This
repo intentionally leaves out side experiments that were tried during
writing but didn't make it into the final post.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Everything downloads its own data on first run (WikiText-103 and the HF
tokenizers via `datasets`/`transformers`, cached by HF after the first call;
no full model weights are downloaded, only tokenizer configs).

Static frequency tables are checked in alongside the scripts that need them
so you don't have to re-tokenize the full WikiText-103 train split (~118M
tokens) just to load a probability table. Delete a `static_probs_*.npy` file
and re-run its script if you want to regenerate it from scratch instead.

## Layout

Each directory is one section of the blog post, runnable independently:

### `01_storage_efficiency/` — the main benchmark chart

- `bench_static.py` — static-table ANS vs per-document ANS (the ~900B/doc
  per-doc-table overhead argument for why the table has to be shared), plus
  a block-size sweep (does training a fresh table per 16KB/1MB block ever
  beat one shared static table? No, not until multi-MB blocks).
- `bench_static_full.py` — full run across length buckets, r50k + cl100k,
  gives the min/median/max numbers in the post's results table
  (r50k+ANS: 1.68x/3.37x/4.30x, cl100k+ANS: 1.68x/3.30x/4.14x).
- `bench_competitors.py` — the byte-codec baselines (gzip/zstd/brotli) and
  the `zstd --train` dictionary comparison (2.61x, "the fairest competitor
  to r50k+ANS"), plus the order-0 byte-level ANS check (1.73x) that isolates
  how much of the ratio is the tokenizer vs. the entropy coder.

### `02_multi_tokenizer/` — "this isn't an OpenAI-specific trick"

- `bench_cl100k_packing.py` — cl100k packing widths: uint32 (current), 3-byte
  (practical fix), and the exact-minimum 17-bit packing (better ratio, much
  costlier decode, why 3-byte wins in practice).
- `bench_o200k_full.py` — same methodology for o200k (GPT-4o's tokenizer):
  ratio (1.14x uint32 / 1.52x 3-byte / 3.32x ANS) and full latency
  (tokenizer-only / +3-byte / +ANS, median and p99).
- `bench_other_llms.py` — Qwen2.5, DeepSeek-V2, Gemma-2 (real HF tokenizers,
  tokenizer config only, no model weights): confirms the same packing/ANS
  pattern holds industry-wide, and that Gemma-2's 256,000-token vocabulary
  (the largest tested) gets the best compression of the six.

### `03_latency/` — encode/decode latency benchmark

- `bench_latency.py` / `bench_latency_static.py` — median + p99 latency for
  every method in the storage benchmark, split into tokenizer-only vs
  +ANS increments (the numbers behind the stacked latency chart).

### `04_frequency_remap/` — the cheaper alternative to ANS

- `freq_remap_full.py` — reassigns token IDs by real corpus frequency rank
  instead of BPE's merge-discovery order, then compares uint32 / uint16 or
  3-byte (whichever is practical) / frequency-remap+streamvbyte / ANS. This
  is the experiment behind "A Faster Alternative to ANS" and the actual ask
  to tokenizer vendors: assign IDs by frequency at training time and this
  entire section becomes free.

### `05_mxbai_wordpiece/` — do embedding-model tokenizers compress well too?

- `bench_mxbai_percentiles.py` — mxbai-embed-large-v1's WordPiece tokenizer
  (30,522-token vocab) vs r50k: nearly identical raw packing, ~5% better
  compression with ANS, but an 80%+ character error rate from BERT's
  lowercasing (`"Qdrant"` → `"qdrant"`), which is why it's not a viable
  lossless codec despite the better ratio.

## Reproducing specific numbers

| Blog claim | Script |
| --- | --- |
| r50k raw uint16 already beats every byte codec (2.33x) | `01_storage_efficiency/bench_static_full.py` |
| r50k + static ANS reaches 3.37x | `01_storage_efficiency/bench_static_full.py` |
| Per-document tables cost ~900 bytes and erase the gain | `01_storage_efficiency/bench_static.py` |
| The fairest competitor, `zstd --train`, reaches 2.61x | `01_storage_efficiency/bench_competitors.py` |
| How much of the ratio is the tokenizer vs. the entropy coder (order-0 byte ANS: 1.73x) | `01_storage_efficiency/bench_competitors.py` |
| cl100k/o200k naive uint32 packing is worse than gzip; 3-byte packing fixes it | `02_multi_tokenizer/bench_cl100k_packing.py`, `bench_o200k_full.py` |
| Same pattern holds for Qwen2.5 / DeepSeek-V2 / Gemma-2 | `02_multi_tokenizer/bench_other_llms.py` |
| ANS decode is slower than LZ4, encode/decode latency table | `03_latency/bench_latency.py`, `bench_latency_static.py` |
| Frequency-sorted IDs + streamvbyte recovers most of ANS's ratio at ~1/15th the decode latency | `04_frequency_remap/freq_remap_full.py` |
| mxbai WordPiece: comparable ratio, 80%+ CER from lowercasing | `05_mxbai_wordpiece/bench_mxbai_percentiles.py` |

## License

MIT. Cite the blog post if you use these numbers elsewhere.
