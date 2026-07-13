"""
Benchmark: static ANS vs trained-dictionary competitors.

Competitors:
  - zstd --train (dictionary trained on WikiText-103 train split)
  - brotli (built-in web-optimized static dictionary)
  - gzip -9, zstd -22 (no dict) — baselines

All evaluated on WikiText-103 test split (out-of-sample).
Also tested on the reference RAG doc (domain mismatch test).
"""

import gzip, os
import zstandard as zstd
import brotli
import tiktoken
import numpy as np
import constriction
from collections import Counter, defaultdict
from datasets import load_dataset

TOKENIZERS = {"r50k": ("r50k_base", 50_257), "cl100k": ("cl100k_base", 100_277)}
CACHE_DIR = os.path.dirname(__file__)
LENGTH_BUCKETS = [(0, 75), (75, 125), (125, 175), (175, 250), (250, 350), (350, 600)]
DICT_SIZES = [32 * 1024, 64 * 1024, 112 * 1024]  # 32KB, 64KB, 112KB

REF_DOC = """Retrieval-Augmented Generation (RAG) is a technique that combines the strengths
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
frequency distribution of tokens in natural language corpora.""".strip()

# ── load static ANS models (cached) ──────────────────────────────────────────

def load_static_probs(name, enc_name, vocab_size):
    cache = os.path.join(CACHE_DIR, f"static_probs_{name}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    raise FileNotFoundError(f"Run bench_static_full.py first to build {cache}")

print("=== Loading static ANS models ===")
static_models, encs = {}, {}
for name, (enc_name, vocab_size) in TOKENIZERS.items():
    probs = load_static_probs(name, enc_name, vocab_size)
    static_models[name] = constriction.stream.model.Categorical(probs, perfect=False)
    encs[name] = tiktoken.get_encoding(enc_name)
    print(f"  {name}: loaded")

def ans_encode(ids, model):
    enc = constriction.stream.stack.AnsCoder()
    enc.encode_reverse(np.array(ids, dtype=np.int32), model)
    return len(enc.get_compressed().tobytes())

# ── train zstd dictionaries on WikiText-103 train split ───────────────────────

def dict_cache_path(sz): return os.path.join(CACHE_DIR, f"zstd_dict_{sz//1024}k.bin")

all_cached = all(os.path.exists(dict_cache_path(sz)) for sz in DICT_SIZES)
if all_cached:
    print("\n=== Loading cached zstd dictionaries ===")
    zstd_dicts = {}
    for sz in DICT_SIZES:
        with open(dict_cache_path(sz), "rb") as f:
            raw = f.read()
        zstd_dicts[sz] = zstd.ZstdCompressionDict(raw)
        print(f"  {sz//1024}KB dict: loaded ({len(raw):,} bytes)")
else:
    print("\n=== Training zstd dictionaries on WikiText-103 train split ===")
    ds_train = load_dataset("wikitext", "wikitext-103-v1", split="train", trust_remote_code=True)
    train_articles = [t.strip().encode("utf-8") for t in ds_train["text"]
                      if len(t.strip().split()) >= 30]
    print(f"  Train articles (>=30 words): {len(train_articles):,}")
    samples = train_articles[:50_000]
    print(f"  Using {len(samples):,} samples for dictionary training")

    zstd_dicts = {}
    for sz in DICT_SIZES:
        print(f"  Training {sz//1024}KB dictionary...")
        d = zstd.train_dictionary(sz, samples)
        raw = d.as_bytes()
        tmp = dict_cache_path(sz) + ".tmp"
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, dict_cache_path(sz))
        zstd_dicts[sz] = d
        print(f"    Done: {len(raw):,} bytes, saved")

# ── build compressors ─────────────────────────────────────────────────────────

zstd_plain = zstd.ZstdCompressor(level=22)
zstd_with_dicts = {sz: zstd.ZstdCompressor(level=22, dict_data=d)
                   for sz, d in zstd_dicts.items()}

def compress_doc(text):
    raw = text.encode("utf-8")
    n = len(raw)
    if n == 0:
        return None

    result = {
        "bytes": n,
        "words": len(text.split()),
        "gzip":       n / len(gzip.compress(raw, compresslevel=9)),
        "zstd":       n / len(zstd_plain.compress(raw)),
        "brotli_q9":  n / len(brotli.compress(raw, quality=9)),
        "brotli_q11": n / len(brotli.compress(raw, quality=11)),
    }
    for sz, c in zstd_with_dicts.items():
        result[f"zstd_dict_{sz//1024}k"] = n / len(c.compress(raw))

    for name, model in static_models.items():
        ids = encs[name].encode(text)
        if not ids:
            return None
        result[f"{name}_static_ans"] = n / ans_encode(ids, model)

    return result

# ── evaluate on test split ────────────────────────────────────────────────────

print("\n=== Evaluating on WikiText-103 test split ===")
ds_test = load_dataset("wikitext", "wikitext-103-v1", split="test", trust_remote_code=True)
articles = [t.strip() for t in ds_test["text"] if len(t.strip().split()) >= 30]
print(f"  Articles: {len(articles):,}")

all_results = []
for text in articles:
    r = compress_doc(text)
    if r:
        all_results.append(r)

def avg(results, key):
    return sum(r[key] for r in results) / len(results)

# ── results: overall ─────────────────────────────────────────────────────────

METHODS = [
    ("gzip -9",          "gzip"),
    ("zstd -22",         "zstd"),
    ("brotli q=9",       "brotli_q9"),
    ("brotli q=11",      "brotli_q11"),
    ("zstd dict 32KB",   "zstd_dict_32k"),
    ("zstd dict 64KB",   "zstd_dict_64k"),
    ("zstd dict 112KB",  "zstd_dict_112k"),
    ("r50k + static ANS","r50k_static_ans"),
    ("cl100k + static ANS","cl100k_static_ans"),
]

W = 72
print(f"\n{'='*W}")
print(f"  WikiText-103 test split — {len(all_results):,} articles")
print(f"{'='*W}")
print(f"  {'Method':<26} {'Overall':>8}  per length bucket →")
print(f"  {'':26} {'avg':>8}  ", end="")
for lo, hi in LENGTH_BUCKETS:
    print(f"{lo+1}-{hi}".center(8), end=" ")
print()
print(f"  {'-'*(W-2)}")

bucket_results = defaultdict(list)
for r in all_results:
    for lo, hi in LENGTH_BUCKETS:
        if lo < r["words"] <= hi:
            bucket_results[(lo, hi)].append(r)

for label, key in METHODS:
    overall = avg(all_results, key)
    print(f"  {label:<26} {overall:>7.2f}x  ", end="")
    for lo, hi in LENGTH_BUCKETS:
        docs = bucket_results[(lo, hi)]
        if docs:
            print(f"{avg(docs, key):>7.2f}x ", end="")
        else:
            print(f"{'—':>8} ", end="")
    print()

bucket_ns = [len(bucket_results[b]) for b in LENGTH_BUCKETS]
print(f"  {'N docs':<26} {len(all_results):>8}  ", end="")
for n in bucket_ns:
    print(f"{n:>7}  ", end="")
print()
print(f"{'='*W}")

# ── reference RAG doc (domain mismatch test) ──────────────────────────────────

print(f"\n{'='*W}")
print(f"  Reference RAG doc ({len(REF_DOC.encode()):,} bytes, {len(REF_DOC.split())} words)")
print(f"  — domain mismatch: technical RAG content vs Wikipedia-trained codecs")
print(f"{'='*W}")
ref = compress_doc(REF_DOC)
for label, key in METHODS:
    print(f"  {label:<26} {ref[key]:>7.2f}x")
print(f"{'='*W}")
