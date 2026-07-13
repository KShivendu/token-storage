"""
How badly does the WikiText-trained static ANS table (used everywhere else in
the post, and in the interactive pipeline widget's "code"/"thai" presets) do
on Python code and Thai text, vs a table actually trained on that domain?

Real corpora:
  - Python: real .py files from this machine's local repos (train/test split
    by file, not by line, so no leakage).
  - Thai: real Thai Wikipedia articles (wikimedia/wikipedia 20231101.th),
    streamed, train/test split by article.

Tested against r50k, cl100k, and o200k tokenizers.
"""

import glob
import math
import random
from collections import Counter

import numpy as np
import tiktoken
import constriction
from datasets import load_dataset

random.seed(0)

TOKENIZERS = {
    "r50k": "r50k_base",
    "cl100k": "cl100k_base",
    "o200k": "o200k_base",
}

# ── build real Python corpus (train/test split by file) ─────────────────────

print("Collecting local .py files...")
py_files = glob.glob("/home/kshivendu/projects/**/*.py", recursive=True)
py_files = [f for f in py_files if "node_modules" not in f and "/.venv/" not in f]
random.shuffle(py_files)

def read_files(paths, char_budget):
    out, total = [], 0
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                t = fh.read()
        except OSError:
            continue
        if not t.strip():
            continue
        out.append(t)
        total += len(t)
        if total >= char_budget:
            break
    return out

py_train_files = py_files[: len(py_files) // 2]
py_test_files = py_files[len(py_files) // 2 :]
py_train_docs = read_files(py_train_files, 8_000_000)
py_test_docs = read_files(py_test_files, 2_000_000)
py_test_docs = [d for d in py_test_docs if 200 <= len(d) <= 20_000][:150]
print(f"  Python train: {sum(len(d) for d in py_train_docs):,} chars, {len(py_train_docs)} files")
print(f"  Python test: {len(py_test_docs)} files")

# ── build real Thai corpus (train/test split by article) ────────────────────

print("\nStreaming Thai Wikipedia...")
ds = load_dataset("wikimedia/wikipedia", "20231101.th", split="train", streaming=True)
thai_articles = []
it = iter(ds)
for _ in range(400):
    try:
        doc = next(it)
    except StopIteration:
        break
    if len(doc["text"]) >= 200:
        thai_articles.append(doc["text"])
random.shuffle(thai_articles)
thai_train_docs = thai_articles[: len(thai_articles) // 2]
thai_test_docs = [d.strip() for d in thai_articles[len(thai_articles) // 2 :] if 200 <= len(d.strip()) <= 20_000][:150]
print(f"  Thai train: {sum(len(d) for d in thai_train_docs):,} chars, {len(thai_train_docs)} articles")
print(f"  Thai test: {len(thai_test_docs)} articles")

# ── helpers ───────────────────────────────────────────────────────────────────

def train_table(enc, docs, vocab_size):
    ids = enc.encode("\n".join(docs), disallowed_special=())
    counts = np.ones(vocab_size, dtype=np.float64)  # Laplace smoothing
    for tok_id, c in Counter(ids).items():
        counts[tok_id] += c
    return counts / counts.sum()

def ans_ratio(enc, probs, docs):
    model = constriction.stream.model.Categorical(probs, perfect=False)
    total_raw = total_compressed = 0
    for text in docs:
        raw = text.encode("utf-8")
        ids = enc.encode(text, disallowed_special=())
        if not ids:
            continue
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(np.array(ids, dtype=np.int32), model)
        compressed = coder.get_compressed().tobytes()
        total_raw += len(raw)
        total_compressed += len(compressed)
    return total_raw / total_compressed

W = 92
print(f"\n{'='*W}")
print(f"  Domain mismatch: WikiText-trained ANS table vs domain-matched table")
print(f"{'='*W}")
print(f"  {'tokenizer':<10} {'domain':<8} {'WikiText-trained':>18} {'domain-matched':>16}")
print(f"  {'-'*(W-2)}")

for name, enc_name in TOKENIZERS.items():
    enc = tiktoken.get_encoding(enc_name)
    vocab_size = enc.n_vocab
    wiki_probs = np.load(f"static_probs_{name}.npy")

    py_probs = train_table(enc, py_train_docs, vocab_size)
    py_wiki_ratio = ans_ratio(enc, wiki_probs, py_test_docs)
    py_matched_ratio = ans_ratio(enc, py_probs, py_test_docs)
    print(f"  {name:<10} {'python':<8} {py_wiki_ratio:>17.3f}x {py_matched_ratio:>15.3f}x")

    thai_probs = train_table(enc, thai_train_docs, vocab_size)
    thai_wiki_ratio = ans_ratio(enc, wiki_probs, thai_test_docs)
    thai_matched_ratio = ans_ratio(enc, thai_probs, thai_test_docs)
    print(f"  {name:<10} {'thai':<8} {thai_wiki_ratio:>17.3f}x {thai_matched_ratio:>15.3f}x")
    print(f"  {'-'*(W-2)}")

print(f"{'='*W}")
