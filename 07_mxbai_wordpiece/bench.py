"""
Compression benchmark: raw text vs gzip/zstd/lz4 vs BPE (r50k) vs mxbai WordPiece.
Includes recovery measurement (exact match + character error rate).
"""

import gzip
import lz4.frame
import zstandard as zstd
import tiktoken
import struct
import math
from collections import Counter
from transformers import AutoTokenizer
import constriction
import numpy as np

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

# ── recovery ──────────────────────────────────────────────────────────────────

import re
import difflib

def _cer(a: str, b: str) -> float:
    matcher = difflib.SequenceMatcher(None, a, b)
    matching = sum(t.size for t in matcher.get_matching_blocks())
    edits = len(a) - matching + abs(len(b) - len(a))
    return edits / len(a) if a else 0.0

def _normalise(s: str) -> str:
    return re.sub(r'\s+', ' ', s.lower()).strip()

def _fix_punct(s: str) -> str:
    s = re.sub(r'\( ', '(', s)
    s = re.sub(r' \)', ')', s)
    s = re.sub(r'(\w) - (\w)', r'\1-\2', s)
    s = re.sub(r' ([.,;:!?])', r'\1', s)
    return s

def measure_recovery(label: str, original: str, token_ids: list, decode_fn, unk_id=None):
    unk_count     = token_ids.count(unk_id) if unk_id is not None else 0
    reconstructed = decode_fn(token_ids)
    orig_n        = _normalise(original)
    recon_n       = _normalise(reconstructed)
    return {
        "label":       label,
        "tokens":      len(token_ids),
        "unk_rate":    unk_count / len(token_ids) if token_ids else 0.0,
        "exact_match": reconstructed == original,
        "cer":         _cer(original, reconstructed),
        "cer_norm":    _cer(orig_n, recon_n),
        "cer_fix":     _cer(orig_n, _fix_punct(recon_n)),
    }

# ── compression helpers ───────────────────────────────────────────────────────

def entropy_lower_bound(symbols: list) -> int:
    counts = Counter(symbols)
    total = len(symbols)
    bits = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return int((bits * total) / 8)

def ans_encode(ids: list) -> bytes:
    counts  = Counter(ids)
    vocab   = sorted(counts.keys())
    v2i     = {v: i for i, v in enumerate(vocab)}
    freqs   = np.array([counts[v] for v in vocab], dtype=np.float64)
    freqs  /= freqs.sum()
    model   = constriction.stream.model.Categorical(freqs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(np.array([v2i[t] for t in ids], dtype=np.int32), model)
    return encoder.get_compressed().tobytes()

def compress_token_ids(ids: list, fmt: str) -> dict:
    raw  = struct.pack(f">{len(ids)}{fmt}", *ids)
    gz   = gzip.compress(raw, compresslevel=9)
    zs   = zstd.ZstdCompressor(level=22).compress(raw)
    l4   = lz4.frame.compress(raw, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)
    ans  = ans_encode(ids)
    return {"raw": raw, "gz": gz, "zstd": zs, "lz4": l4, "ans": ans, "entropy": entropy_lower_bound(ids)}

def fmt_bytes(n: int) -> str:
    return f"{n:>7,} bytes"

def ratio(orig: int, comp: int) -> str:
    return f"{orig/comp:.2f}x"

# ── tokenizers ────────────────────────────────────────────────────────────────

r50k  = tiktoken.get_encoding("r50k_base")
mxtok = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")

raw_bytes = TEXT.encode("utf-8")
original  = len(raw_bytes)

# standard compression on raw text
gz_text   = gzip.compress(raw_bytes, compresslevel=9)
zstd_text = zstd.ZstdCompressor(level=22).compress(raw_bytes)
lz4_text  = lz4.frame.compress(raw_bytes, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)

# r50k BPE (uint16, vocab=50,257)
r50k_ids  = r50k.encode(TEXT)
r50k_c    = compress_token_ids(r50k_ids, "H")
r50k_rec  = measure_recovery("r50k BPE", TEXT, r50k_ids, r50k.decode)

# mxbai WordPiece (uint16, vocab=30,522)
mx_ids    = mxtok.encode(TEXT, add_special_tokens=False)
mx_c      = compress_token_ids(mx_ids, "H")
mx_rec    = measure_recovery("mxbai WordPiece", TEXT, mx_ids,
                              lambda ids: mxtok.decode(ids),
                              unk_id=mxtok.unk_token_id)

# ── print ─────────────────────────────────────────────────────────────────────

W = 138
print(f"\n{'='*W}")
print(f"  Document: {len(TEXT.split())} words, {original:,} bytes raw UTF-8")
print(f"{'='*W}")
print(f"  {'Method':<32} {'Tokens':>7} {'Raw':>10} {'+ gzip':>10} {'+ zstd':>10} {'+ lz4':>10} {'+ ANS':>10} {'Entropy lb':>10} {'CER':>6} {'CER*':>6} {'CER**':>7} {'UNK%':>6}")
print(f"  {'-'*(W-2)}")

def row(label, tokens, raw, gz, zs, l4, ans, entropy, cer=None, cer_norm=None, cer_fix=None, unk=None):
    cer_s   = f"{cer*100:>5.1f}%"      if cer      is not None else f"{'—':>6}"
    cern_s  = f"{cer_norm*100:>5.1f}%" if cer_norm  is not None else f"{'—':>6}"
    cerf_s  = f"{cer_fix*100:>6.1f}%"  if cer_fix   is not None else f"{'—':>7}"
    unk_s   = f"{unk*100:>5.1f}%"      if unk      is not None else f"{'—':>6}"
    tok_s   = f"{tokens:>7}"           if tokens   is not None else f"{'—':>7}"
    l4_s    = f"{l4:>10,}"             if l4       is not None else f"{'—':>10}"
    ans_s   = f"{ans:>10,}"            if ans      is not None else f"{'—':>10}"
    print(f"  {label:<32} {tok_s} {raw:>10,} {gz:>10,} {zs:>10,} {l4_s} {ans_s} {entropy:>10,} {cer_s} {cern_s} {cerf_s} {unk_s}")

def sep():
    print(f"  {'-'*(W-2)}")

row("Raw UTF-8",                None, original, len(gz_text), len(zstd_text), len(lz4_text), None,                original,         cer=0.0, cer_norm=0.0, cer_fix=0.0, unk=0.0)
sep()
row("r50k BPE (uint16)",        r50k_rec['tokens'], len(r50k_c['raw']), len(r50k_c['gz']), len(r50k_c['zstd']), len(r50k_c['lz4']), len(r50k_c['ans']), r50k_c['entropy'], cer=r50k_rec['cer'], cer_norm=r50k_rec['cer_norm'], cer_fix=r50k_rec['cer_fix'], unk=r50k_rec['unk_rate'])
sep()
row("mxbai WordPiece (uint16)", mx_rec['tokens'],   len(mx_c['raw']),   len(mx_c['gz']),   len(mx_c['zstd']),   len(mx_c['lz4']),   len(mx_c['ans']),   mx_c['entropy'],   cer=mx_rec['cer'],   cer_norm=mx_rec['cer_norm'],   cer_fix=mx_rec['cer_fix'],   unk=mx_rec['unk_rate'])

print(f"{'='*W}")
print()
print(f"  Raw        = token IDs packed as uint16, no further compression")
print(f"  Entropy lb = Shannon lower bound (theoretical best lossless compression)")
print(f"  ANS        = Asymmetric Numeral Systems (arithmetic coding via constriction)")
print(f"  CER        = character error rate vs original")
print(f"  CER*       = CER ignoring case + whitespace")
print(f"  CER**      = CER ignoring case + whitespace + punct spacing")
