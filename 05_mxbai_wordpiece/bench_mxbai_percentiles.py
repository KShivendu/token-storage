"""
mxbai WordPiece + static ANS — min/median/max + CER across WikiText-103 test.

Uses the `tokenizers` Rust library directly (not transformers.AutoTokenizer)
for fast batch encoding — same backend, skips Python wrapper overhead.
"""

import os, json
import numpy as np
import constriction
from datasets import load_dataset
from tokenizers import Tokenizer

CACHE_DIR = os.path.dirname(__file__)
VOCAB = 30_522
CACHE_PATH = os.path.join(CACHE_DIR, "static_probs_mxbai.npy")
BATCH = 4096

tok = Tokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
tok.no_truncation()
tok.no_padding()

# ── build static freq table from train split (cached) ─────────────────────────

def build_probs():
    if os.path.exists(CACHE_PATH):
        print("  Loading cached mxbai probs...")
        return np.load(CACHE_PATH)
    print("  Building mxbai freq table from WikiText-103 train...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    texts = [t for t in ds["text"] if t.strip()]
    counts = np.ones(VOCAB, dtype=np.int64)
    for i in range(0, len(texts), BATCH):
        for enc in tok.encode_batch(texts[i:i + BATCH], add_special_tokens=False):
            for tid in enc.ids:
                if 0 <= tid < VOCAB:
                    counts[tid] += 1
        if i % (BATCH * 10) == 0:
            print(f"    {i}/{len(texts)}...")
    probs = counts.astype(np.float64) / counts.sum()
    np.save(CACHE_PATH, probs)
    return probs

probs = build_probs()
model = constriction.stream.model.Categorical(probs, perfect=False)

# ── evaluate on test split ─────────────────────────────────────────────────────

print("Loading WikiText-103 test split...")
ds_test = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test")
articles = [t.strip() for t in ds_test["text"] if len(t.strip().split()) >= 30]
print(f"  {len(articles)} articles (>=30 words)")

import re

def fix_cer(s):
    s = re.sub(r'\( ', '(', s)
    s = re.sub(r' \)', ')', s)
    s = re.sub(r'(\w) - (\w)', r'\1-\2', s)
    return s

def cer(original, decoded):
    n = max(len(original), len(decoded))
    if n == 0:
        return 0.0
    mismatches = sum(a != b for a, b in zip(original, decoded)) + abs(len(original) - len(decoded))
    return mismatches / n * 100

ratios, raw_u16_ratios, cer_raw, cer_fixed = [], [], [], []

for i in range(0, len(articles), BATCH):
    batch = articles[i:i + BATCH]
    encs = tok.encode_batch(batch, add_special_tokens=False)
    for text, enc in zip(batch, encs):
        n = len(text.encode("utf-8"))
        ids = enc.ids
        if not ids:
            continue
        raw_u16_ratios.append(n / (len(ids) * 2))  # mxbai's 30,522 vocab fits uint16
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(np.array(ids, dtype=np.int32), model)
        compressed = coder.get_compressed().tobytes()
        ratios.append(n / len(compressed))
        buf = np.frombuffer(compressed, dtype=np.uint32).copy()
        dec_ids = constriction.stream.stack.AnsCoder(buf).decode(model, len(ids)).tolist()
        dec_text = tok.decode(dec_ids)
        cer_raw.append(cer(text, dec_text))
        cer_fixed.append(cer(text, fix_cer(dec_text)))

# ── results ────────────────────────────────────────────────────────────────────

def summarise_cer(vals, label):
    nonzero = sum(1 for v in vals if v > 0)
    print(f"  {label:<16} avg={sum(vals)/len(vals):.1f}%  "
          f"median={float(np.median(vals)):.1f}%  "
          f"articles with CER>0: {nonzero}/{len(vals)}")

print(f"\n{'='*64}")
print(f"  mxbai WordPiece — {len(ratios)} articles")
print(f"{'='*64}")
print(f"  raw uint16:   min={min(raw_u16_ratios):.2f}x  median={np.median(raw_u16_ratios):.2f}x  max={max(raw_u16_ratios):.2f}x")
print(f"  + static ANS: min={min(ratios):.2f}x  median={np.median(ratios):.2f}x  max={max(ratios):.2f}x")
summarise_cer(cer_raw,   "CER raw:")
summarise_cer(cer_fixed, "CER after fix:")
print(f"{'='*64}")

result = {
    "raw_u16_min": round(min(raw_u16_ratios), 2),
    "raw_u16_median": round(float(np.median(raw_u16_ratios)), 2),
    "raw_u16_max": round(max(raw_u16_ratios), 2),
    "min": round(min(ratios), 2),
    "median": round(float(np.median(ratios)), 2),
    "max": round(max(ratios), 2),
    "cer_raw_avg_pct": round(sum(cer_raw) / len(cer_raw), 2),
    "cer_fixed_avg_pct": round(sum(cer_fixed) / len(cer_fixed), 2),
    "cer_fixed_nonzero": sum(1 for v in cer_fixed if v > 0),
    "n_articles": len(ratios),
}
with open(os.path.join(CACHE_DIR, "bench_mxbai_results.json"), "w") as f:
    json.dump(result, f, indent=2)
print("Saved to bench_mxbai_results.json")
