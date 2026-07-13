"""
Raw (no-ANS) fixed-width packing ratio on the same real Python and Thai test
corpora used in domain_mismatch_python_thai.py (same seed, same construction,
so results are directly comparable to the ANS numbers already in the post).
"""
import glob
import random

import tiktoken
from datasets import load_dataset

random.seed(0)

TOKENIZERS = {
    "r50k": ("r50k_base", 2),      # 2 bytes/token (uint16)
    "cl100k": ("cl100k_base", 3),  # 3-byte packing (24-bit)
    "o200k": ("o200k_base", 3),
}

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

py_test_files = py_files[len(py_files) // 2 :]
py_test_docs = read_files(py_test_files, 2_000_000)
py_test_docs = [d for d in py_test_docs if 200 <= len(d) <= 20_000][:150]

print("Streaming Thai Wikipedia...")
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
thai_test_docs = [d.strip() for d in thai_articles[len(thai_articles) // 2 :] if 200 <= len(d.strip()) <= 20_000][:150]

print(f"Python test: {len(py_test_docs)} files, Thai test: {len(thai_test_docs)} articles\n")

def raw_ratio(enc, bytes_per_token, docs):
    total_raw = total_packed = 0
    for text in docs:
        raw = text.encode("utf-8")
        ids = enc.encode(text, disallowed_special=())
        if not ids:
            continue
        total_raw += len(raw)
        total_packed += len(ids) * bytes_per_token
    return total_raw / total_packed

W = 70
print("=" * W)
print("  Raw fixed-width packing ratio (no ANS)")
print("=" * W)
print(f"  {'tokenizer':<10} {'domain':<8} {'ratio':>10}")
print(f"  {'-'*(W-2)}")
for name, (enc_name, bpt) in TOKENIZERS.items():
    enc = tiktoken.get_encoding(enc_name)
    py_r = raw_ratio(enc, bpt, py_test_docs)
    thai_r = raw_ratio(enc, bpt, thai_test_docs)
    print(f"  {name:<10} {'python':<8} {py_r:>9.3f}x")
    print(f"  {name:<10} {'thai':<8} {thai_r:>9.3f}x")
print("=" * W)
