"""
Domain mismatch on real Hindi Wikipedia text, cl100k and o200k only (same
methodology as domain_mismatch_python_thai.py): WikiText-trained ANS table
vs a table trained on real Hindi Wikipedia articles (streamed, train/test
split by article, no leakage).
"""
import random
from collections import Counter

import numpy as np
import tiktoken
import constriction
from datasets import load_dataset

random.seed(0)

TOKENIZERS = {
    "cl100k": "cl100k_base",
    "o200k": "o200k_base",
}

print("Streaming Hindi Wikipedia...")
ds = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
hindi_articles = []
it = iter(ds)
for _ in range(400):
    try:
        doc = next(it)
    except StopIteration:
        break
    if len(doc["text"]) >= 200:
        hindi_articles.append(doc["text"])
random.shuffle(hindi_articles)
hindi_train_docs = hindi_articles[: len(hindi_articles) // 2]
hindi_test_docs = [d.strip() for d in hindi_articles[len(hindi_articles) // 2 :] if 200 <= len(d.strip()) <= 20_000][:150]
print(f"  Hindi train: {sum(len(d) for d in hindi_train_docs):,} chars, {len(hindi_train_docs)} articles")
print(f"  Hindi test: {len(hindi_test_docs)} articles")

def train_table(enc, docs, vocab_size):
    ids = enc.encode("\n".join(docs), disallowed_special=())
    counts = np.ones(vocab_size, dtype=np.float64)
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

def raw_ratio(enc, docs):
    total_raw = total_packed = 0
    for text in docs:
        raw = text.encode("utf-8")
        ids = enc.encode(text, disallowed_special=())
        if not ids:
            continue
        total_raw += len(raw)
        total_packed += len(ids) * 3  # 3-byte packing (both cl100k/o200k fit in 24 bits)
    return total_raw / total_packed

W = 92
print(f"\n{'='*W}")
print("  Hindi domain mismatch: WikiText-trained ANS table vs Hindi-matched table")
print(f"{'='*W}")
print(f"  {'tokenizer':<10} {'raw (no ANS)':>14} {'WikiText-trained':>18} {'Hindi-matched':>15}")
print(f"  {'-'*(W-2)}")

for name, enc_name in TOKENIZERS.items():
    enc = tiktoken.get_encoding(enc_name)
    vocab_size = enc.n_vocab
    wiki_probs = np.load(f"static_probs_{name}.npy")

    raw_r = raw_ratio(enc, hindi_test_docs)
    hindi_probs = train_table(enc, hindi_train_docs, vocab_size)
    wiki_ratio = ans_ratio(enc, wiki_probs, hindi_test_docs)
    matched_ratio = ans_ratio(enc, hindi_probs, hindi_test_docs)
    print(f"  {name:<10} {raw_r:>13.3f}x {wiki_ratio:>17.3f}x {matched_ratio:>14.3f}x")

print(f"{'='*W}")
