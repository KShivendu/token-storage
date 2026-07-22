# token-storage

Real, runnable experiments behind [Token-Native Storage: Read and Write in the
Language Models Already Speak](https://www.kshivendu.dev/blog/token-storage).

Every number quoted in that post comes from one of the scripts below, run
against real tokenizers (`tiktoken`'s r50k/cl100k/o200k, HuggingFace's
Qwen2.5/DeepSeek-V2/Gemma-2/BERT-WordPiece), the real `constriction` ANS
library, and real held-out corpora: English ([C4](https://huggingface.co/datasets/allenai/c4)),
code ([codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean),
Python), and Hindi ([Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)).
Nothing here is estimated or simulated.

Note: the scripts internally call the English/prose domain `"prose"` — that's
the same thing the post calls `"english"`.

## Shared conventions

Unless a script says otherwise:

- **512-token chunks** (the post's main chart also sweeps 256/512/2000 tokens —
  see `bench_summary_tables.py`).
- **40 sampled test chunks** per domain, drawn from a held-out test split that
  the training-side frequency/dictionary tables never see.
- **RNG seed 3344** for most scripts. Exception: `bench_summary_tables.py`
  and `bench_freqremap.py` use seed **9012** — different seed, same
  methodology (40 held-out test chunks, train/test split never mixed). Both
  seeds are real seeds used to produce numbers that are actually in the post;
  there isn't a single universal seed across the whole repo.
- **Bootstrap confidence intervals** (2000 resamples, 90% CI) around the
  median for ratio/latency claims, so a single unlucky chunk can't swing a
  reported number.
- Latency for fast operations (sub-few-µs: unpacking, ANS/streamvbyte
  decode, rank remap/lookup) is the **median of 30 back-to-back reps per
  chunk** — a single `perf_counter()` pair is dominated by GC/scheduler
  noise at that scale. Slower operations (tokenize, real compress/decompress)
  are timed single-shot per chunk, which is fine at 100s of µs.
- All methods are **lossless** except mxbai WordPiece (see `05_mxbai_wordpiece/`).

This repo intentionally leaves out earlier/exploratory scripts once a later
script fixed a real methodology bug in them or reran the same measurement
more rigorously (see "Superseded scripts" at the bottom) and anything from an
unrelated side-project (an FSE/tANS/KV-cache-compression experiment series)
that happened to live in the same working directory during benchmarking.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Then build the corpus once (downloads C4/codeparrot/Hindi-Wikipedia via
`datasets`, streamed rather than fully downloaded, and pre-tokenizes them
with r50k):

```bash
python 00_corpus_prep/prep_multidomain.py
```

This writes `data/corpus/{prose,code,hindi}_{train,test}.npy` (compact
r50k token-id arrays, not raw text — a few hundred MB total). Every other
script reads from `data/corpus/`, so this must run first. `05_mxbai_wordpiece/`
is the one exception: it downloads WikiText-103 directly (see below for why).

## Layout

Each directory is one section of the blog post, runnable independently once
`data/corpus/` exists:

### `00_corpus_prep/` — builds the shared corpus

- `prep_multidomain.py` — streams C4 (en), codeparrot-clean (Python), and
  Hindi Wikipedia from HuggingFace, tokenizes with r50k, and splits each into
  a TRAIN portion (~8M tokens, used only to build frequency/ANS/zstd-dict
  tables) and a held-out TEST portion (~1.5M tokens, used only for
  measurement). Nothing downstream ever trains on the test split.

### `01_storage_efficiency/` — the main benchmark chart + full results table

- `bench_summary_tables.py` — the interactive chart's `ratioValues`/latency
  numbers for **10 of the 15 methods per domain** (everything except the
  three `+freq` bars), swept across all **three chunk sizes (256/512/2000
  tokens)** and all three domains — this is the "follow-up post" reference
  in the post's commented-out chunk-size aside. Also supports `--zstd-dict`
  and `--lz4-blocks` flags for two extra baselines (ES/Lucene-style block
  LZ4, and the zstd `--train` dictionary ratio at each chunk size) merged
  into the same `summary_tables_256_512_2000.json`. Reproduces the chart
  exactly for Code, Hindi, and English at 256/2000 tokens. For English/512,
  the chart's 10 non-`+freq` ratios instead come from
  `bench_full_results_table.py` (see below). Note: this script's RNG
  is shared and consumed sequentially across the `for chunk_size in
  CHUNK_SIZES: for domain in DOMAINS` loop — running only a slice of it
  (e.g. isolating just `chunk_size=512`) desyncs the random stream and will
  NOT reproduce the same sampled chunks; run the whole script.
- `bench_freqremap.py` — the `+freq` bars in the main chart (all 3 domains ×
  all 3 tokenizers, all 3 chunk sizes) — this is the **only** script in the
  repo that computes `+freq` compression ratio across more than one domain,
  and it's the actual source of the main chart's `+freq` values, verified by
  running it: English/512 r50k+freq 2.62x, cl100k+freq 2.71x, o200k+freq
  2.76x; code/512 1.42x/2.57x/2.53x; hindi/512 1.33x/2.04x/4.39x, all exact
  matches. Its per-component *timing* breakdown was superseded by
  `bench_freq_split.py` (see below), but its *ratio* numbers are not
  duplicated anywhere else and are load-bearing for the main chart.
- `bench_full_results_table.py` — the "Full results and latency tables"
  collapsible: min/median/max compression ratio and median/p99 latency,
  English (C4), 512-token chunks, keeping per-chunk arrays (so min/max/p99
  are real, not derived).
- `bench_zstd_train_fix.py` — fixes a real bug in an earlier zstd `--train`
  timing measurement (a fresh `ZstdCompressor` was created per call, forcing
  the 112KB trained dictionary to be re-digested every time: ~22.6ms/call
  vs ~28µs/call once precomputed and reused). Feeds the `zstdT` compress/
  decompress numbers in the post's `latParts`.
- `bench_entropy_vs_tokenizer.py` — the "entropy coder vs. tokenizer" and
  "streamvbyte vs. +freq rank-remap contribution" collapsible: ANS on raw
  UTF-8 bytes (1.76x) vs. token-level ANS (3.26x) vs. zstd-19 on packed
  token-ID bytes (2.73x); and streamvbyte with vs. without rank-remap
  (2.13x vs 2.64x).
- `check_id_vs_freq.py` — optional/supporting: the Spearman correlation
  check behind "BPE assigns token IDs in merge-discovery order, not
  frequency" (a qualitative claim in the post's prose, not a specific
  number quoted in a chart/table — kept because it's the actual evidence
  for that claim, not required to reproduce any single figure).

### `02_multi_tokenizer/` — "this isn't an OpenAI-specific trick"

- `bench_tokenizer_gen.py` — the cross-tokenizer table: r50k, cl100k,
  o200k, Qwen2.5, DeepSeek-V2, Gemma-2 (real HF tokenizer configs, no model
  weights downloaded), each with uint32/16 raw / 3-byte raw / +static-ANS
  ratios, on the same English (C4), 512-token-chunk methodology as the
  rest of the post.

### `03_latency/` — the Agent/Human write & read latency charts

- `bench_agent_writer.py` — write-side components when the writer is an
  agent (source is already token IDs): `detokenize_only` (what byte codecs
  pay), `raw_pack_only`, `freq_encode_only`, `ans_encode_only`.
- `bench_agent_mode_v2.py` — read-side components for both Agent mode
  (decompress+tokenize for byte codecs; unpack/ANS-decode only for
  token-native, no detokenize) and the byte-codec `decompress_only`
  baseline. This is `v2`: it fixes a real bug in `bench_agent_mode.py`
  (single-shot timing on sub-microsecond ops is noise-dominated — e.g. the
  original had r50k unpack measuring *slower* than LZ4 decompress, which
  makes no sense for a zero-copy reinterpret vs. real decompression). `v2`
  times fast ops as a median of 30 reps instead. Together with
  `bench_agent_writer.py` and `bench_freq_split.py` (below), these three
  scripts populate every value in the post's `latParts`.

### `04_frequency_remap/` — the `+freq` method: ratio and latency breakdown

- `bench_freqremap.py` — the `+freq` compression ratio for the main chart
  (see above), across all 3 domains, all 3 tokenizers, all 3 chunk sizes.
- `bench_freq_split.py` — splits `+freq`'s encode/decode into its real
  components (an earlier script — `bench_freqremap.py` above — only timed
  the streamvbyte codec calls, with the numpy rank-remap/lookup step done
  outside the timed region): `rank_remap_only`, `streamvbyte_enc`,
  `streamvbyte_dec`, `rank_lookup_only`, per domain and tokenizer. Feeds
  `rankRemap`/`svbEnc`/`svbDec`/`rankLookup` in `latParts`. (Only the timing
  methodology in `bench_freqremap.py` is superseded here — its ratio numbers,
  above, are still the canonical source for the main chart.)

### `05_mxbai_wordpiece/` — do embedding-model tokenizers compress well too?

- `bench_mxbai_percentiles.py` — mxbai-embed-large-v1's WordPiece tokenizer
  (30,522-token vocab): raw uint16 (median 2.35x) and +static ANS (median
  3.56x), plus the character-error-rate (CER) analysis (avg 80.4%) that's
  the reason WordPiece isn't viable as a lossless codec. **Uses WikiText-103
  test articles, not C4** — the post's own CER claim is specifically
  "80.4% of WikiText-103 articles came back corrupted," so this script's
  corpus choice matches the post's actual wording, even though the rest of
  the repo has moved to C4/codeparrot/Hindi-Wikipedia. Ships with a cached
  `static_probs_mxbai.npy` so you don't need to rebuild the frequency table
  from scratch. Verified to still reproduce the post's numbers exactly.

## Reproducing specific numbers

| Blog claim | Script |
| --- | --- |
| Main chart: LZ4/gzip/zstd/brotli/raw/+ANS bars, all domains, 512-token chunks | `01_storage_efficiency/bench_summary_tables.py` (default run) |
| Main chart: `zstd --train` bars, all domains, 512-token chunks | `01_storage_efficiency/bench_summary_tables.py --zstd-dict` (verified exact: prose 2.66x, code 3.24x, hindi 4.56x) |
| Main chart: `+freq` bars, all domains, 512-token chunks | `04_frequency_remap/bench_freqremap.py` |
| 256/512/2000-token chunk-size sweep | all three of the above, run at each `chunk_size` in `CHUNK_SIZES` |
| "Full results" min/median/max ratio + latency table (English, 512-tok) | `01_storage_efficiency/bench_full_results_table.py` |
| zstd `--train` compress/decompress latency (`latParts.zstdT`) | `01_storage_efficiency/bench_zstd_train_fix.py` |
| Entropy coder vs. tokenizer contribution (1.76x / 3.26x / 2.73x) | `01_storage_efficiency/bench_entropy_vs_tokenizer.py` |
| `+freq` ratio contribution: rank-remap vs. streamvbyte alone (2.13x / 2.64x) | `01_storage_efficiency/bench_entropy_vs_tokenizer.py` |
| "BPE assigns IDs by merge order, not frequency" (Spearman check) | `01_storage_efficiency/check_id_vs_freq.py` |
| Cross-tokenizer table (r50k/cl100k/o200k/Qwen2.5/DeepSeek-V2/Gemma-2) | `02_multi_tokenizer/bench_tokenizer_gen.py` |
| Write latency chart, Agent mode (`latParts.rawPack/rankRemap/svbEnc/ansEnc`, detokenize tax for byte codecs) | `03_latency/bench_agent_writer.py` |
| Read latency chart, Agent mode (`latParts.rawUnpack/ansDec`, tokenize tax for byte codecs) | `03_latency/bench_agent_mode_v2.py` |
| `+freq` component split (`latParts.rankRemap/svbEnc/svbDec/rankLookup`) | `04_frequency_remap/bench_freq_split.py` |
| mxbai WordPiece: comparable ratio, 80.4% CER from lowercasing | `05_mxbai_wordpiece/bench_mxbai_percentiles.py` |

## Superseded scripts (not included)

**A note on corpus history:** an earlier version of this repo (and an earlier
version of the post) benchmarked everything on WikiText-103 only. The post
was later rewritten to use three held-out domains (C4/codeparrot/Hindi
Wikipedia) instead, and the numbers changed as a result (e.g. the old
WikiText-103 run reported r50k+ANS median 3.37x; the current post's own
"Full results" table, on C4, reports 3.26x). The old `01_storage_efficiency`
/ `02_multi_tokenizer` / `03_latency` / `04_frequency_remap` scripts from
that earlier version have been fully replaced by the ones listed above —
none of the old scripts' numbers match what's in the post today. Only
`05_mxbai_wordpiece/` was untouched, since the post's own WordPiece/CER
claim is specifically about WikiText-103.

Beyond that corpus migration, some individual scripts were real earlier runs
during the current writing process, superseded by a later script that fixed
a measurement bug or reran the same thing more rigorously — kept out of this
repo so there's exactly one script per number:

- `bench_agent_mode.py` → superseded by `bench_agent_mode_v2.py` (fixed
  single-shot timing noise on sub-microsecond ops).
- `bench_agent_serving.py` → an earlier, narrower version of the Agent/Human
  latency split (only zstd-19 as the representative byte codec); superseded
  by `bench_agent_writer.py` + `bench_agent_mode_v2.py`, which cover every
  codec.
- `bench_full_results_table.py` vs. `bench_summary_tables.py`: both kept,
  different collapsibles. For English/512, the chart uses
  `bench_full_results_table.py`'s ratios (more accurate at n=40 chunks);
  `bench_summary_tables.py` remains the source everywhere else.

Also excluded: everything that belongs to a separate, unrelated FSE/tANS/
KV-cache-compression project (tracked in a different repo, `axiom-labs/wholembed`)
that happened to share a working directory on the benchmark host
(`ablation_table_size.py`, `bench_long_context.py`, `bench_real_fse.py`,
`bench_tans.py`, `tans.py`, `bench_multidomain.py`, `bench_raw_and_tokenizers.py`,
`test_zstd_dict_overhead*.py`, `fse_src/`, `libfse_wrap.so`, `wrapper.c`) —
none of these produce a number that appears in this post.

## License

MIT. Cite the blog post if you use these numbers elsewhere.
