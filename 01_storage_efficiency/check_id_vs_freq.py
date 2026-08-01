import os
import sys
import numpy as np
import tiktoken
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import load_ids

r50k = tiktoken.get_encoding("r50k_base")
VOCAB = 50257

train_ids = load_ids("prose_train")
train_text = r50k.decode(train_ids[:400*512].tolist())
train_tok_ids = r50k.encode(train_text, disallowed_special=())

counts = np.zeros(VOCAB, dtype=np.int64)
vals, cnts = np.unique(train_tok_ids, return_counts=True)
counts[vals] = cnts

seen = counts > 0
ids_seen = np.arange(VOCAB)[seen]
counts_seen = counts[seen]

rho, p = spearmanr(ids_seen, -counts_seen)  # correlate raw ID with frequency (negated so high freq = low value)
print(f"Spearman corr(raw token ID, frequency rank) = {rho:.3f} (p={p:.2e})")
print(f"n tokens seen in train corpus: {seen.sum()} / {VOCAB}")

order = np.argsort(-counts)
rank_of = np.empty(VOCAB, dtype=np.int64)
rank_of[order] = np.arange(VOCAB)

print("\nTop 15 most frequent tokens: (raw_id, rank, count, repr)")
for tid in order[:15]:
    print(f"  raw_id={tid:6d}  rank={rank_of[tid]:6d}  count={counts[tid]:6d}  repr={repr(r50k.decode([tid]))}")

print("\nHow far are the 1000 most frequent tokens' raw IDs spread out?")
top1000_ids = order[:1000]
print(f"  min raw_id={top1000_ids.min()}, max raw_id={top1000_ids.max()}, median raw_id={np.median(top1000_ids):.0f}")

# fraction of bottom-1000-by-ID that are also in top-1000-by-frequency
bottom1000_by_id = np.arange(1000)
overlap = len(set(bottom1000_by_id.tolist()) & set(top1000_ids.tolist()))
print(f"  overlap between lowest-1000-IDs and most-frequent-1000-tokens: {overlap}/1000")
