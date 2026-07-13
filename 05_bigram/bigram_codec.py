"""
Real bigram ANS codec: pruned (prev,cur) pairs with absolute-discount backoff
to the static unigram table. Matches the 1.4MB/175K-pairs budget already
cited in the blog's "How Far Can Token Compression Go?" table (4.35x).
"""
import os, time, math
import numpy as np
import tiktoken
import constriction
from datasets import load_dataset

VOCAB = 50257
D = 0.75
N_KEEP = 175_000  # matches the "1MB" (really 1.4MB @ 8B/pair) budget

enc = tiktoken.get_encoding("r50k_base")

CACHE = "bigram_table_r50k.npz"
if os.path.exists(CACHE):
    print("Loading cached bigram table...")
    d = np.load(CACHE)
    kept_prev, kept_cur, kept_prob, alpha, backnorm, P_uni = (
        d["kept_prev"], d["kept_cur"], d["kept_prob"], d["alpha"], d["backnorm"], d["P_uni"]
    )
else:
    print("Tokenizing WikiText-103 train split...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train", trust_remote_code=True)
    ids = enc.encode_ordinary("\n".join(ds["text"]))
    ids = np.array(ids, dtype=np.int64)
    print(f"  {len(ids):,} train tokens")

    counts = np.bincount(ids, minlength=VOCAB).astype(np.int64) + 1
    P_uni = (counts.astype(np.float64) / counts.sum())

    keys = ids[:-1] * VOCAB + ids[1:]
    K, C = np.unique(keys, return_counts=True)
    prev_all = (K // VOCAB).astype(np.int32)
    cur_all = (K % VOCAB).astype(np.int32)
    N_prev = np.bincount(prev_all, weights=C.astype(np.float64), minlength=VOCAB)
    print(f"  {len(K):,} unique pairs")

    order_desc = np.argsort(-C, kind="stable")
    keep_idx = order_desc[:N_KEEP]
    kept_prev = prev_all[keep_idx]
    kept_cur = cur_all[keep_idx]
    kept_C = C[keep_idx].astype(np.float64)

    safe_N = np.maximum(N_prev, 1.0)
    kept_prob = (kept_C - D) / safe_N[kept_prev]  # discounted P(cur|prev) for kept pairs

    kept_mass = np.bincount(kept_prev, weights=kept_C - D, minlength=VOCAB)
    uni_kept_mass = np.zeros(VOCAB)
    np.add.at(uni_kept_mass, kept_prev, P_uni[kept_cur])
    alpha = np.maximum(np.where(N_prev > 0, 1.0 - kept_mass / safe_N, 1.0), 1e-12)
    backnorm = np.maximum(1.0 - uni_kept_mass, 1e-12)

    np.savez(CACHE, kept_prev=kept_prev, kept_cur=kept_cur, kept_prob=kept_prob,
             alpha=alpha, backnorm=backnorm, P_uni=P_uni)
    print(f"Saved bigram table cache ({os.path.getsize(CACHE)/1024/1024:.2f} MB on disk, "
          f"conceptual budget {N_KEEP*8/1024/1024:.2f} MB @ 8B/pair)")

# ── build a fast per-prev-token lookup: prev -> {cur: prob} ──────────────────
from collections import defaultdict
print("Building per-prev-token lookup dict...")
pair_probs = defaultdict(dict)
for p, c, pr in zip(kept_prev.tolist(), kept_cur.tolist(), kept_prob.tolist()):
    pair_probs[p][c] = pr

def dist_for_prev(prev_tok):
    """Full VOCAB-length probability vector for P(cur | prev_tok), with backoff."""
    dist = (alpha[prev_tok] / backnorm[prev_tok]) * P_uni
    overrides = pair_probs.get(prev_tok)
    if overrides:
        for c, pr in overrides.items():
            dist[c] = pr
    dist = dist / dist.sum()  # renormalize for float safety
    return dist

print("Bigram model ready.")
