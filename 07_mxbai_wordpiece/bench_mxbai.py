"""
Compression benchmark adding mixedbread tokenizer alongside r50k_base.
"""
import gzip
import struct
import math
from collections import Counter
import lz4.frame
import zstandard as zstd
import tiktoken
from transformers import AutoTokenizer

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

def entropy_lower_bound(symbols):
    counts = Counter(symbols)
    total = len(symbols)
    bits = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return int((bits * total) / 8)

def compress_all(token_ids, pack_fmt, pack_size):
    raw = struct.pack(f">{len(token_ids)}{pack_fmt}", *token_ids)
    gz   = gzip.compress(raw, compresslevel=9)
    zs   = zstd.ZstdCompressor(level=22).compress(raw)
    l4   = lz4.frame.compress(raw, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)
    return raw, gz, zs, l4

def fmt(n):    return f"{n:>7,} bytes"
def ratio(orig, comp): return f"{orig/comp:.2f}x"

raw_bytes = TEXT.encode("utf-8")
original  = len(raw_bytes)

gz_text   = gzip.compress(raw_bytes, compresslevel=9)
zstd_text = zstd.ZstdCompressor(level=22).compress(raw_bytes)
lz4_text  = lz4.frame.compress(raw_bytes, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)

# r50k_base (GPT-2 BPE, vocab=50,257, fits uint16)
r50k      = tiktoken.get_encoding("r50k_base")
r50k_ids  = r50k.encode(TEXT)
r50k_raw, r50k_gz, r50k_zstd, r50k_lz4 = compress_all(r50k_ids, "H", 2)
r50k_entropy = entropy_lower_bound(r50k_ids)

# mixedbread (WordPiece, vocab=30,522, fits uint16)
mxtok     = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
mx_ids    = mxtok.encode(TEXT, add_special_tokens=False)
mx_unk    = mx_ids.count(mxtok.unk_token_id)
mx_raw, mx_gz, mx_zstd, mx_lz4 = compress_all(mx_ids, "H", 2)
mx_entropy = entropy_lower_bound(mx_ids)

print(f"\n{'='*62}")
print(f"  Document: {len(TEXT.split())} words")
print(f"{'='*62}")
print(f"  {'Method':<38} {'Size':>10}  {'Ratio':>6}")
print(f"  {'-'*56}")
print(f"  {'Raw UTF-8 text':<38} {fmt(original)}  {'1.00x':>6}")
print(f"  {'gzip -9':<38} {fmt(len(gz_text))}  {ratio(original, len(gz_text)):>6}")
print(f"  {'zstd -22':<38} {fmt(len(zstd_text))}  {ratio(original, len(zstd_text)):>6}")
print(f"  {'lz4 max':<38} {fmt(len(lz4_text))}  {ratio(original, len(lz4_text)):>6}")
print(f"  {'-'*56}")
print(f"  {'r50k BPE uint16 (raw)':<38} {fmt(len(r50k_raw))}  {ratio(original, len(r50k_raw)):>6}   ({len(r50k_ids)} tokens)")
print(f"  {'r50k BPE + gzip':<38} {fmt(len(r50k_gz))}  {ratio(original, len(r50k_gz)):>6}")
print(f"  {'r50k BPE + zstd':<38} {fmt(len(r50k_zstd))}  {ratio(original, len(r50k_zstd)):>6}")
print(f"  {'r50k BPE + lz4 max':<38} {fmt(len(r50k_lz4))}  {ratio(original, len(r50k_lz4)):>6}")
print(f"  {'r50k entropy lower bound':<38} {fmt(r50k_entropy)}  {ratio(original, r50k_entropy):>6}")
print(f"  {'-'*56}")
unk_note = f" *** {mx_unk} UNK tokens ***" if mx_unk else " (no UNK)"
print(f"  {'mxbai WordPiece uint16 (raw)':<38} {fmt(len(mx_raw))}  {ratio(original, len(mx_raw)):>6}   ({len(mx_ids)} tokens){unk_note}")
print(f"  {'mxbai WordPiece + gzip':<38} {fmt(len(mx_gz))}  {ratio(original, len(mx_gz)):>6}")
print(f"  {'mxbai WordPiece + zstd':<38} {fmt(len(mx_zstd))}  {ratio(original, len(mx_zstd)):>6}")
print(f"  {'mxbai WordPiece + lz4 max':<38} {fmt(len(mx_lz4))}  {ratio(original, len(mx_lz4)):>6}")
print(f"  {'mxbai entropy lower bound':<38} {fmt(mx_entropy)}  {ratio(original, mx_entropy):>6}")
print(f"{'='*62}")
