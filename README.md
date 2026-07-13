# token-storage

Real, runnable experiments behind [Token-Native Storage: Read and Write in the
Language Models Already Speak](https://www.kshivendu.dev/blog/token-storage).

Every number quoted in that post comes from one of the scripts below, run
against real tokenizers (`tiktoken`'s r50k/cl100k/o200k, HuggingFace's
Qwen2.5/DeepSeek-V2/Gemma-2/mxbai), the real `constriction` ANS library, and
real corpora (WikiText-103, GitHub Archive, Hindi Wikipedia, local Python
source files). Nothing here is estimated or simulated.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Everything downloads its own data on first run (WikiText-103 and the HF
tokenizers via `datasets`/`transformers`, both cached by HF after the first
call). The one exception is `06_domain_mismatch/`, see below.

Static frequency tables and the pruned bigram table are checked into
`cache/` (and copied alongside the scripts that need them) so you don't have
to re-tokenize the full WikiText-103 train split (~118M tokens) just to load
a probability table. Delete a `static_probs_*.npy` file and re-run its
script if you want to regenerate it from scratch instead.

## Layout

Each directory is one section of the blog post(s), runnable independently:

### `01_storage_efficiency/` — main storage benchmark

The headline chart: r50k/cl100k raw packing vs LZ4/gzip/zstd/brotli vs
static ANS, across 1,773 WikiText-103 test articles.

- `bench_static.py` — the original benchmark: static-table ANS vs per-document
  ANS (and the ~900B/doc per-doc-table overhead argument), plus a block-size
  sweep (does training a fresh table per 16KB/1MB block ever beat one shared
  static table? No, not until multi-MB blocks).
- `bench_static_full.py` — full run across length buckets, r50k + cl100k,
  gives the min/median/max numbers in the post's results table
  (r50k+ANS: 1.68x/3.37x/4.30x, cl100k+ANS: 1.68x/3.30x/4.14x).
- `bench_competitors.py` — the byte-codec baselines and `zstd --train`
  dictionary comparison (2.61x, "the fairest competitor to r50k+ANS").

### `02_multi_tokenizer/` — "this isn't an OpenAI-specific trick"

- `bench_cl100k_packing.py` — cl100k packing widths: uint32 (current), 3-byte
  (practical fix), and the exact-minimum 17-bit packing (better ratio, much
  costlier decode, why 3-byte wins in practice).
- `bench_o200k_full.py` — same methodology for o200k (GPT-4o's tokenizer):
  ratio (1.14x uint32 / 1.52x 3-byte / 3.32x ANS) and full latency
  (tokenizer-only / +3-byte / +ANS, median and p99).
- `bench_other_llms.py` — Qwen2.5, DeepSeek-V2, Gemma-2 (real HF tokenizers,
  no model weights downloaded, tokenizer config only): confirms the same
  packing/ANS pattern holds industry-wide, and that Gemma-2's 256,000-token
  vocabulary (the largest tested) gets the best compression of the six.

### `03_latency/` — encode/decode latency benchmark

- `bench_latency.py` / `bench_latency_static.py` — median + p99 latency for
  every method in the storage benchmark, split into tokenizer-only vs
  +ANS increments (the numbers behind the stacked latency chart).

### `04_frequency_remap/` — the cheaper alternative to ANS

- `freq_remap_full.py` — reassigns token IDs by real corpus frequency rank
  instead of BPE's merge-discovery order, then compares uint32 / uint64 /
  practical (uint16 or 3-byte) / frequency-remap+streamvbyte / ANS. This is
  the experiment behind "A Faster Alternative to ANS" and the actual ask to
  tokenizer vendors: assign IDs by frequency at training time and this
  entire section becomes free.

### `05_bigram/` — how far compression goes with more context

- `expA_bigram.py` — analytic order-1 (bigram) achievable-ratio sweep across
  table-size budgets (runs on [Modal](https://modal.com), bigram counting
  over ~118M tokens is heavy; `modal run expA_bigram.py`). Produces the
  ~4.35x-at-1.4MB / ~4.86x-at-full-table numbers.
- `bigram_codec.py` — trains a real pruned bigram+backoff table (absolute
  discounting) and caches it to `bigram_table_r50k.npz`.
- `bigram_bench.py` — a real, roundtrip-verified bigram ANS codec (not just
  the analytic estimate): measures actual ratio (4.09x on held-out articles)
  and real per-symbol latency, with model caching per distinct `prev_tok` so
  the latency number reflects the bigram model's real cost, not Python
  object-construction overhead.
- `bigram_json_mismatch.py` — applies the WikiText-trained bigram table to
  real GitHub Archive JSON, showing bigram tables inherit the same
  domain-mismatch sensitivity as unigram ANS (no free lunch from context).

### `06_domain_mismatch/` — what happens off-corpus

- `bench_percentiles.py` — WikiText-103 vs real GitHub Archive JSON: does a
  JSON-trained ANS table beat gzip where the WikiText-trained one loses?
  Needs a GH Archive sample; run `fetch_gharchive_sample.py` first (downloads
  one real public hourly dump, ~70MB, not committed to this repo).
- `fetch_hindi_corpus.py` — lightweight Hindi Wikipedia REST API fetcher (a
  few hundred small article fetches, not a full dataset download).
- `domain_mismatch_python_thai.py` / `domain_mismatch_hindi.py` — real
  domain-mismatch experiments: WikiText-trained vs domain-matched ANS tables
  on real local Python source files and real Hindi Wikipedia text, across
  r50k/cl100k/o200k. This is where the "o200k handles non-English scripts
  much better out of the box" finding comes from (Hindi: o200k 2.62x
  mismatched → 5.74x domain-matched, the best ratio in either post).
- `raw_packing_python_thai.py` — the no-ANS-at-all baseline for the same
  Python/Thai text, isolating how much of the domain-mismatch story is the
  entropy coder vs. raw tokenizer behavior.

### `07_mxbai_wordpiece/` — do embedding-model tokenizers compress well too?

- `bench.py` — general compression + lossless-recovery harness (exact match
  + character error rate) for BPE vs WordPiece.
- `bench_mxbai.py` / `bench_mxbai_percentiles.py` — mxbai-embed-large-v1's
  WordPiece tokenizer (30,522-token vocab) vs r50k: nearly identical raw
  packing, ~5% better compression with ANS, but an 80%+ character error rate
  from BERT's lowercasing (`"Qdrant"` → `"qdrant"`), which is why it's not
  a viable lossless codec despite the better ratio.

### `08_junk_detection/`

- `expB_junk_filter.py` — a free byproduct of ANS encoding:
  the encoder's per-token `-log2 P(token)` gives every document a
  bits/token score for free, and that score separates clean WikiText prose
  from base64/hex/keyboard-mash/spam/HTML junk with AUC 1.0, where a
  zstd-ratio threshold gets the direction backwards on repetitive junk.

## Reproducing specific numbers

| Blog claim | Script |
| --- | --- |
| r50k raw uint16 already beats every byte codec (2.33x) | `01_storage_efficiency/bench_static_full.py` |
| r50k + static ANS reaches 3.37x | `01_storage_efficiency/bench_static_full.py` |
| cl100k/o200k naive uint32 packing is worse than gzip; 3-byte packing fixes it | `02_multi_tokenizer/bench_cl100k_packing.py`, `bench_o200k_full.py` |
| Same pattern holds for Qwen2.5 / DeepSeek-V2 / Gemma-2 | `02_multi_tokenizer/bench_other_llms.py` |
| ANS decode is slower than LZ4, encode/decode latency table | `03_latency/bench_latency.py` |
| Frequency-sorted IDs + streamvbyte recovers ~78-87% of ANS's ratio at ~1/15th the decode latency | `04_frequency_remap/freq_remap_full.py` |
| Bigram table: ~4.35x analytic, 4.09x real coded, same domain-mismatch sensitivity as unigram | `05_bigram/expA_bigram.py`, `bigram_bench.py`, `bigram_json_mismatch.py` |
| WikiText-trained table on JSON: 1.10x vs JSON-trained: 2.24x | `06_domain_mismatch/bench_percentiles.py` |
| o200k handles Hindi far better than cl100k out of the box; domain-matched o200k on Hindi hits 5.74x | `06_domain_mismatch/domain_mismatch_hindi.py` |
| mxbai WordPiece: comparable ratio, 80%+ CER from lowercasing | `07_mxbai_wordpiece/bench_mxbai_percentiles.py` |
| Per-token entropy as a free out-of-distribution / junk detector (AUC 1.0) | `08_junk_detection/expB_junk_filter.py` |

## A known gap

The blog's frequency-remap+varint JSON-mismatch numbers (1.08x
WikiText-trained / 1.99x JSON-trained) were run ad hoc during writing and
the script wasn't saved. `04_frequency_remap/freq_remap_full.py` reproduces
the remap methodology on WikiText-103; adapting it to the GH Archive JSON
sample from `06_domain_mismatch/` would reproduce that specific result.

## License

MIT. Cite the blog posts if you use these numbers elsewhere.
