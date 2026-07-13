import gzip, json
import numpy as np
import tiktoken
import constriction
from collections import defaultdict

VOCAB = 50257
enc = tiktoken.get_encoding("r50k_base")

d = np.load("bigram_table_r50k.npz")
kept_prev, kept_cur, kept_prob = d["kept_prev"], d["kept_cur"], d["kept_prob"]
alpha, backnorm, P_uni = d["alpha"], d["backnorm"], d["P_uni"]

pair_probs = defaultdict(dict)
for p, c, pr in zip(kept_prev.tolist(), kept_cur.tolist(), kept_prob.tolist()):
    pair_probs[p][c] = pr

def dist_for_prev(prev_tok):
    dist = (alpha[prev_tok] / backnorm[prev_tok]) * P_uni
    overrides = pair_probs.get(prev_tok)
    if overrides:
        for c, pr in overrides.items():
            dist[c] = pr
    return dist / dist.sum()

def bigram_ans_encode(ids):
    coder = constriction.stream.stack.AnsCoder()
    for i in range(len(ids) - 1, -1, -1):
        if i == 0:
            model = constriction.stream.model.Categorical(P_uni, perfect=False)
        else:
            model = constriction.stream.model.Categorical(dist_for_prev(int(ids[i-1])), perfect=False)
        coder.encode_reverse(np.array([ids[i]], dtype=np.int32), model)
    return coder.get_compressed().tobytes()

print("Loading JSON events (cached GitHub Archive sample)...")
json_events = []
with gzip.open("gharchive_sample.json.gz", "rt", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.strip()
        if not line: continue
        try:
            json_events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
        if len(json_events) >= 100:  # small sample, per-symbol loop is slow
            break
json_docs = [json.dumps(e, separators=(",", ":")) for e in json_events]
print(f"{len(json_docs)} JSON docs")

total_raw = total_bigram = 0
for text in json_docs:
    raw = text.encode("utf-8")
    ids = enc.encode(text)
    if not ids: continue
    total_raw += len(raw)
    total_bigram += len(bigram_ans_encode(ids))

print(f"\nWikiText-trained bigram model applied to JSON:")
print(f"  ratio: {total_raw/total_bigram:.3f}x")
print(f"  (reference: unigram ANS, WikiText-trained on JSON: 1.10x; JSON-trained: 2.24x)")
print(f"  (reference: freq-remap+varint, WikiText-trained on JSON: 1.08x; JSON-trained: 1.99x)")
