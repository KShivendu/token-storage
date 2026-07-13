"""
Static corpus ANS benchmark over real WikiText-103 articles.

- Static probs trained on WikiText-103 train split (cached)
- Evaluated on WikiText-103 test split (out-of-sample)
- Groups docs by length bucket, reports mean compression ratio per bucket
- Compares r50k + static ANS, cl100k + static ANS, gzip, zstd
"""

import gzip, os
import zstandard as zstd
import tiktoken
import numpy as np
import constriction
from collections import Counter, defaultdict
from datasets import load_dataset

TOKENIZERS = {"r50k": ("r50k_base", 50_257), "cl100k": ("cl100k_base", 100_277)}
CACHE_DIR = os.path.dirname(__file__)
LENGTH_BUCKETS = [(0, 75), (75, 125), (125, 175), (175, 250), (250, 350), (350, 600)]

# ── static freq tables (train split, cached) ──────────────────────────────────

def load_static_probs(name, enc_name, vocab_size):
    cache = os.path.join(CACHE_DIR, f"static_probs_{name}.npy")
    if os.path.exists(cache):
        print(f"  {name}: loading cached probs from {cache}")
        return np.load(cache)
    print(f"  {name}: tokenizing WikiText-103 train split...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train", trust_remote_code=True)
    enc = tiktoken.get_encoding(enc_name)
    ids = enc.encode("\n".join(ds["text"]))
    counts = np.ones(vocab_size, dtype=np.int64)
    for tok_id, cnt in Counter(ids).items():
        counts[tok_id] += cnt
    probs = counts.astype(np.float64) / counts.sum()
    np.save(cache, probs)
    return probs

print("=== Static frequency tables ===")
static_probs, encs, static_models = {}, {}, {}
for name, (enc_name, vocab_size) in TOKENIZERS.items():
    static_probs[name] = load_static_probs(name, enc_name, vocab_size)
    encs[name] = tiktoken.get_encoding(enc_name)
    static_models[name] = constriction.stream.model.Categorical(static_probs[name], perfect=False)

# ── helpers ───────────────────────────────────────────────────────────────────

zstd_c = zstd.ZstdCompressor(level=22)

def ans_encode(ids, model):
    enc = constriction.stream.stack.AnsCoder()
    enc.encode_reverse(np.array(ids, dtype=np.int32), model)
    return len(enc.get_compressed().tobytes())

def compress_doc(text):
    raw = text.encode("utf-8")
    n = len(raw)
    if n == 0:
        return None
    gz  = len(gzip.compress(raw, compresslevel=9))
    zs  = len(zstd_c.compress(raw))
    r50k_ids  = encs["r50k"].encode(text)
    cl100k_ids = encs["cl100k"].encode(text)
    if not r50k_ids:
        return None
    r50k_ans  = ans_encode(r50k_ids,  static_models["r50k"])
    cl100k_ans = ans_encode(cl100k_ids, static_models["cl100k"])
    words = len(text.split())
    return {"bytes": n, "words": words,
            "gzip": n/gz, "zstd": n/zs,
            "r50k_static": n/r50k_ans, "cl100k_static": n/cl100k_ans}

# ── evaluate on test split ────────────────────────────────────────────────────

print("\n=== Evaluating on WikiText-103 test split ===")
ds_test = load_dataset("wikitext", "wikitext-103-v1", split="test", trust_remote_code=True)

# collect full articles (skip short lines / headers)
articles = [t.strip() for t in ds_test["text"] if len(t.strip().split()) >= 30]
print(f"  Articles with >=30 words: {len(articles)}")

buckets = defaultdict(list)
for text in articles:
    r = compress_doc(text)
    if r is None:
        continue
    w = r["words"]
    for lo, hi in LENGTH_BUCKETS:
        if lo < w <= hi:
            buckets[(lo, hi)].append(r)
            break

# ── results ───────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  Compression ratio by document length  (WikiText-103 test, static ANS)")
print(f"{'='*80}")
print(f"  {'Words':>12}  {'N':>5}  {'gzip':>6}  {'zstd':>6}  {'r50k+ANS':>10}  {'cl100k+ANS':>12}")
print(f"  {'-'*70}")

all_results = []
for (lo, hi) in LENGTH_BUCKETS:
    docs = buckets[(lo, hi)]
    if not docs:
        print(f"  {lo+1:>5}–{hi:<5}      0  (no docs)")
        continue
    def avg(k): return sum(d[k] for d in docs) / len(docs)
    print(f"  {lo+1:>5}–{hi:<5}  {len(docs):>5}  "
          f"{avg('gzip'):>5.2f}x  {avg('zstd'):>5.2f}x  "
          f"{avg('r50k_static'):>9.2f}x  {avg('cl100k_static'):>11.2f}x")
    all_results.extend(docs)

print(f"  {'-'*70}")
if all_results:
    def avg_all(k): return sum(d[k] for d in all_results) / len(all_results)
    print(f"  {'Overall':>12}  {len(all_results):>5}  "
          f"{avg_all('gzip'):>5.2f}x  {avg_all('zstd'):>5.2f}x  "
          f"{avg_all('r50k_static'):>9.2f}x  {avg_all('cl100k_static'):>11.2f}x")
print(f"{'='*80}")
print(f"\n  Note: static ANS table trained on WikiText-103 train split.")
print(f"  No per-doc freq table needed — codec is fully self-contained.")
