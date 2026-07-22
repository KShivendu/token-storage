"""
Build a larger, multi-domain, honestly-split corpus for the tANS/FSE/rANS/
streamvbyte comparison, replacing the WikiText-103-only setup (which gave
us as few as ~7 independent chunks at 8k-token length -- not reliable).

Three domains, streamed (not fully downloaded) from HuggingFace:
  - prose: allenai/c4, en
  - code:  codeparrot/github-code-clean (python)
  - hindi: allenai/mc4, hi

For each domain: stream documents, tokenize with r50k, split into a TRAIN
portion (for the static frequency table -- large) and a TEST portion (held
out, never touched by the table -- used only for eval). Saves compact
token-id arrays (uint32 .npy) per domain/split, not raw text, so disk stays
small (a few hundred MB total for tens of millions of tokens).
"""
import os
import numpy as np
import tiktoken
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_TOKENS_TARGET = 8_000_000
TEST_TOKENS_TARGET = 1_500_000

enc = tiktoken.get_encoding("r50k_base")


def tokenize_stream(ds_iter, train_target, test_target, text_key="text"):
    train_ids, test_ids = [], []
    train_docs, test_docs = 0, 0
    for ex in ds_iter:
        text = ex[text_key]
        if not text or len(text) < 200:
            continue
        ids = enc.encode(text, disallowed_special=())
        if not ids:
            continue
        if len(train_ids) < train_target:
            train_ids.extend(ids)
            train_docs += 1
        elif len(test_ids) < test_target:
            test_ids.extend(ids)
            test_docs += 1
        else:
            break
    return (
        np.array(train_ids, dtype=np.uint32),
        np.array(test_ids, dtype=np.uint32),
        train_docs,
        test_docs,
    )


def save_domain(name, train_arr, test_arr, train_docs, test_docs):
    np.save(os.path.join(OUT_DIR, f"{name}_train.npy"), train_arr)
    np.save(os.path.join(OUT_DIR, f"{name}_test.npy"), test_arr)
    print(
        f"  {name}: train={len(train_arr):,} tok ({train_docs} docs), "
        f"test={len(test_arr):,} tok ({test_docs} docs)"
    )


print("=== prose: allenai/c4 (en) ===")
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
train_arr, test_arr, td, ted = tokenize_stream(iter(ds), TRAIN_TOKENS_TARGET, TEST_TOKENS_TARGET)
save_domain("prose", train_arr, test_arr, td, ted)

print("=== code: codeparrot/codeparrot-clean (Python) ===")
ds = load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")
train_arr, test_arr, td, ted = tokenize_stream(iter(ds), TRAIN_TOKENS_TARGET, TEST_TOKENS_TARGET, text_key="content")
save_domain("code", train_arr, test_arr, td, ted)

print("=== hindi: wikimedia/wikipedia (20231101.hi) ===")
ds = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
train_arr, test_arr, td, ted = tokenize_stream(iter(ds), TRAIN_TOKENS_TARGET, TEST_TOKENS_TARGET)
save_domain("hindi", train_arr, test_arr, td, ted)

print("\nDone. Saved to", OUT_DIR)
