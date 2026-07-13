"""
Static corpus ANS: train token frequencies on WikiText-103,
encode the reference doc with that static table, compare to per-doc ANS.

Key question: does static corpus ANS beat per-doc ANS once you account
for the freq table overhead that per-doc ANS must store?
"""

import struct
import math
from collections import Counter

import numpy as np
import tiktoken
import constriction
from datasets import load_dataset

# ── reference doc (same as blog benchmark) ───────────────────────────────────

TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that combines the strengths
of large language models with external knowledge retrieval. Instead of relying
solely on parametric memory encoded in model weights during training, RAG systems
retrieve relevant documents from a knowledge base at inference time and condition
the model's response on that retrieved context.

The core idea is simple: given a user query, a retrieval component (typically a
dense vector search over a large corpus) fetches the top-k most relevant passages.
These passages are then prepended to the prompt, giving the language model access
to up-to-date, specific, or domain-specific information it may not have seen
during training.

Vector databases like Qdrant, Weaviate, and Pinecone have emerged as the primary
infrastructure layer for RAG systems. They store dense embeddings of documents
alongside the raw text payload, enabling fast approximate nearest-neighbor search
over millions or billions of vectors. The raw text is stored verbatim so it can
be returned to the model as context after retrieval.

A key inefficiency in current RAG deployments is the storage of raw text payloads.
Most vector databases store the original document text as-is, without any
compression. For large corpora this represents a significant storage overhead,
particularly when the same text must be replicated across multiple shards or
replicas for availability. Tokenization-based compression offers a promising
alternative: instead of storing UTF-8 bytes, documents can be stored as sequences
of BPE token IDs, which typically achieve 3x compression over raw ASCII text.
Further gains are possible by applying entropy coding (arithmetic coding or
Huffman coding) over the token ID sequence, exploiting the highly non-uniform
frequency distribution of tokens in natural language corpora.
""".strip()

TOKENIZERS = {
    "r50k":   ("r50k_base",   50_257),
    "cl100k": ("cl100k_base", 100_277),
}

# ── build static freq tables from WikiText-103 ────────────────────────────────

print("Loading WikiText-103 (train split)...")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train", trust_remote_code=True)
corpus_text = "\n".join(ds["text"])
print(f"  Corpus: {len(corpus_text):,} chars")

static_probs_by_enc = {}
doc_ids_by_enc = {}

for name, (enc_name, vocab_size) in TOKENIZERS.items():
    enc = tiktoken.get_encoding(enc_name)
    print(f"\nTokenizing corpus with {name} ({enc_name})...")
    corpus_ids = enc.encode(corpus_text)
    print(f"  Corpus tokens: {len(corpus_ids):,}")
    counts = Counter(corpus_ids)
    print(f"  Unique tokens seen: {len(counts):,} / {vocab_size:,}")

    counts_arr = np.ones(vocab_size, dtype=np.int64)  # Laplace smoothing
    for tok_id, count in counts.items():
        counts_arr[tok_id] += count
    static_probs_by_enc[name] = (enc, vocab_size, counts_arr.astype(np.float64) / counts_arr.sum())
    doc_ids_by_enc[name] = enc.encode(TEXT)

# ── helpers ───────────────────────────────────────────────────────────────────

def entropy_bits(ids, probs):
    """Expected bits using a given probability model."""
    return sum(-math.log2(probs[t]) for t in ids)

def ans_encode_static(ids, probs):
    """ANS encode using pre-computed static probability table."""
    model = constriction.stream.model.Categorical(probs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(np.array(ids, dtype=np.int32), model)
    compressed = encoder.get_compressed().tobytes()

    # Verify round-trip
    decoder = constriction.stream.stack.AnsCoder(
        np.frombuffer(compressed, dtype=np.uint32).copy()
    )
    decoded = decoder.decode(model, len(ids)).tolist()
    assert decoded == ids, "Static ANS round-trip failed!"

    return compressed

def ans_encode_perdoc(ids):
    """Per-doc ANS (current blog approach) — freq table from document only."""
    counts = Counter(ids)
    vocab = sorted(counts.keys())
    v2i = {v: i for i, v in enumerate(vocab)}
    freqs = np.array([counts[v] for v in vocab], dtype=np.float64)
    freqs /= freqs.sum()

    model = constriction.stream.model.Categorical(freqs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(np.array([v2i[t] for t in ids], dtype=np.int32), model)
    compressed = encoder.get_compressed().tobytes()

    # Verify round-trip (decoder gets same local freq table in same session)
    decoder = constriction.stream.stack.AnsCoder(
        np.frombuffer(compressed, dtype=np.uint32).copy()
    )
    decoded_local = decoder.decode(model, len(ids)).tolist()
    decoded_ids = [vocab[i] for i in decoded_local]
    assert decoded_ids == ids, "Per-doc ANS round-trip failed!"

    return compressed, vocab, counts

def freq_table_overhead(vocab, counts):
    """
    Estimate bytes needed to store per-doc freq table alongside the payload.
    Compact encoding: sorted (token_id: uint16, count: uint16) pairs.
    Real systems might compress this further, but this is the baseline.
    """
    n = len(vocab)
    # 2 bytes per token ID (uint16) + 2 bytes per count (uint16)
    return n * 4

# ── encode reference doc ──────────────────────────────────────────────────────

raw_bytes = len(TEXT.encode("utf-8"))
print(f"\nReference doc: {len(TEXT.split()):,} words, {raw_bytes:,} bytes raw UTF-8")

W = 88
print(f"\n{'='*W}")
print(f"  ANS comparison: static corpus table vs per-doc table")
print(f"{'='*W}")
print(f"  {'Method':<44} {'Bytes':>7} {'vs raw':>7}  note")
print(f"  {'-'*(W-2)}")

def row(label, size, note=""):
    print(f"  {label:<44} {size:>7,} {raw_bytes/size:>6.1f}x  {note}")

row("Raw UTF-8", raw_bytes)

for name, (enc, vocab_size, static_probs) in static_probs_by_enc.items():
    doc_ids = doc_ids_by_enc[name]
    id_fmt = "H" if vocab_size <= 65535 else "I"
    raw_token_bytes = struct.pack(f">{len(doc_ids)}{id_fmt}", *doc_ids)

    static_compressed = ans_encode_static(doc_ids, static_probs)
    static_entropy_lb = int(entropy_bits(doc_ids, static_probs) / 8)
    perdoc_compressed, perdoc_vocab, perdoc_counts = ans_encode_perdoc(doc_ids)
    perdoc_overhead = freq_table_overhead(perdoc_vocab, perdoc_counts)

    print(f"  {'-'*(W-2)}")
    row(f"{name} raw packed ({'uint32' if id_fmt == 'I' else 'uint16'})", len(raw_token_bytes))
    row(f"{name} per-doc ANS (payload only)",   len(perdoc_compressed), "← what blog measures")
    row(f"{name} per-doc ANS + freq table",     len(perdoc_compressed) + perdoc_overhead,
        f"overhead: {perdoc_overhead}B ({len(perdoc_vocab)} unique tokens)")
    row(f"{name} static corpus ANS",            len(static_compressed), "self-contained")
    row(f"{name} static entropy lower bound",   static_entropy_lb, "theoretical best w/ static probs")

print(f"{'='*W}")

# ── block-size sweep: at what block size does per-block training break even? ──
#
# LZ4 gets a real win from sharing a compression window across a block of
# documents (ES-style). Does the same trick work for ANS if you train a fresh
# frequency table per block instead of per document? Sweep block size (in raw
# UTF-8 bytes, same units as the ES 16KB-block comparison) and compare:
#   - per-block ANS: fresh table trained ONLY on that block's tokens, table
#     shipped alongside the compressed payload (freq_table_overhead)
#   - static corpus ANS: the same shared WikiText-103-train table used
#     everywhere else in this post, zero marginal overhead per block

print("\nLoading WikiText-103 test split for block-size sweep...")
ds_test = load_dataset("wikitext", "wikitext-103-v1", split="test", trust_remote_code=True)
test_articles = [t.strip() for t in ds_test["text"] if len(t.strip().split()) >= 30]
print(f"  {len(test_articles):,} test articles (>=30 words)")

BLOCK_TARGET_BYTES = [4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000, 1_000_000]

def make_blocks(articles, target_bytes):
    """Concatenate articles up to ~target_bytes, yield one token-id list per block."""
    blocks, cur_text, cur_bytes = [], [], 0
    for a in articles:
        cur_text.append(a)
        cur_bytes += len(a.encode("utf-8"))
        if cur_bytes >= target_bytes:
            blocks.append("\n".join(cur_text))
            cur_text, cur_bytes = [], 0
    return blocks

print(f"\n{'='*W}")
print(f"  Block-size sweep: per-block-trained ANS vs static corpus ANS (r50k)")
print(f"{'='*W}")
print(f"  {'block size':>12}  {'blocks':>7}  {'per-block total':>17}  {'static ANS':>11}  {'ratio':>7}  {'winner':>10}")
print(f"  {'-'*(W-2)}")

enc, vocab_size, static_probs = static_probs_by_enc["r50k"]
for target in BLOCK_TARGET_BYTES:
    blocks = make_blocks(test_articles, target)
    if not blocks:
        continue
    n_blocks = min(len(blocks), 20)  # cap for runtime
    perblock_totals, static_totals = [], []
    for block_text in blocks[:n_blocks]:
        ids = enc.encode(block_text)
        static_c = ans_encode_static(ids, static_probs)
        perdoc_c, perdoc_vocab, perdoc_counts = ans_encode_perdoc(ids)
        overhead = freq_table_overhead(perdoc_vocab, perdoc_counts)
        perblock_totals.append(len(perdoc_c) + overhead)
        static_totals.append(len(static_c))
    avg_perblock = sum(perblock_totals) / len(perblock_totals)
    avg_static = sum(static_totals) / len(static_totals)
    ratio = avg_perblock / avg_static
    winner = "per-block" if ratio < 1.0 else "static"
    print(f"  {target:>10,}B  {n_blocks:>7}  {avg_perblock:>16,.0f}B  {avg_static:>10,.0f}B  {ratio:>6.3f}x  {winner:>10}")

print(f"{'='*W}")
