# token-storage

Experiments behind [Token-Native Storage](https://www.kshivendu.dev/blog/token-storage): does storing text as BPE token IDs instead of UTF-8 bytes get you free compression and lower agent read/write latency?

## Method

- **Corpora**: English ([C4](https://huggingface.co/datasets/allenai/c4)), code ([codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean)), Hindi ([Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)) — held-out train/test splits, nothing trains on test.
- **Tokenizers**: r50k, cl100k, o200k (`tiktoken`), Qwen2.5, DeepSeek-V2, Gemma-2, BERT-WordPiece (HF configs).
- **Chunk sizes**: 512 tokens (main results), also swept at 256/2000.
- 40 sampled test chunks/domain, bootstrap CI (2000 resamples), median-of-30-reps for sub-µs latencies.
- Everything lossless except WordPiece (see findings).

## Findings

| Question | Result |
| --- | --- |
| Compression vs. raw UTF-8 (English, 512-tok, median) | LZ4 1.28x, gzip 1.95x, zstd 1.97x, brotli 2.60x, zstd --train 2.69x, r50k raw 2.27x, r50k +ANS 3.26x |
| Does it hold across tokenizers/vendors? | Yes — r50k/cl100k/o200k/Qwen2.5/DeepSeek-V2/Gemma-2 all land 3.12–3.27x with static ANS |
| Where does the gain come from, coder or tokenizer? | Tokenizer: ANS on raw bytes only reaches 1.76x; feeding it tokens instead of bytes gets to 3.26x |
| Cheaper alternative to ANS (frequency-remap + streamvbyte) | ~13–20% less ratio (English), ~15–20x faster decode |
| Write latency, agent as writer vs. human | 2–20x faster — the writer already produced token IDs, byte codecs pay a re-encode tax |
| Read latency, agent as reader vs. human | Token IDs skip tokenization entirely; byte codecs pay ~450µs/read tokenizing text back |
| Do embedding-model tokenizers (WordPiece) compress too? | Yes, slightly better ratio, but 80% char-error-rate on decode (BERT lowercasing) — not lossless |

Full breakdowns, all three domains, all three chunk sizes: see `01_storage_efficiency/`, `02_multi_tokenizer/`, `03_latency/`, `04_frequency_remap/`, `05_mxbai_wordpiece/`.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python 00_corpus_prep/prep_multidomain.py  # builds data/corpus/, run once
```

Each numbered directory is runnable independently after that and maps to one section of the post.

## License

MIT. Cite the blog post if you use these numbers elsewhere.
