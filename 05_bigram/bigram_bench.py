import time, math, gzip, json
import numpy as np
import tiktoken
import constriction
from collections import defaultdict
from datasets import load_dataset

VOCAB = 50257
enc = tiktoken.get_encoding("r50k_base")

print("Loading bigram table...")
d = np.load("bigram_table_r50k.npz")
kept_prev, kept_cur, kept_prob = d["kept_prev"], d["kept_cur"], d["kept_prob"]
alpha, backnorm, P_uni = d["alpha"], d["backnorm"], d["P_uni"]

pair_probs = defaultdict(dict)
for p, c, pr in zip(kept_prev.tolist(), kept_cur.tolist(), kept_prob.tolist()):
    pair_probs[p][c] = pr

# Model construction is the expensive part (builds a Categorical over a
# 50,257-symbol alphabet). A prev_tok's distribution is deterministic, so
# cache the constructed model per distinct prev_tok instead of rebuilding
# it on every occurrence (a short article reuses the same prev_tok many
# times: ~150 tokens but often <100 distinct token IDs).
model_cache = {}
uni_model = constriction.stream.model.Categorical(P_uni, perfect=False)

def model_for_prev(prev_tok):
    m = model_cache.get(prev_tok)
    if m is None:
        dist = (alpha[prev_tok] / backnorm[prev_tok]) * P_uni
        overrides = pair_probs.get(prev_tok)
        if overrides:
            for c, pr in overrides.items():
                dist[c] = pr
        dist = dist / dist.sum()
        m = constriction.stream.model.Categorical(dist, perfect=False)
        model_cache[prev_tok] = m
    return m

def bigram_ans_encode(ids):
    """Real ANS encode: first token via unigram, rest via bigram+backoff, context-dependent."""
    coder = constriction.stream.stack.AnsCoder()
    # encode in reverse (last token first), each step needs the distribution
    # conditioned on the PREVIOUS token in original order
    for i in range(len(ids) - 1, -1, -1):
        model = uni_model if i == 0 else model_for_prev(int(ids[i-1]))
        coder.encode_reverse(np.array([ids[i]], dtype=np.int32), model)
    return coder.get_compressed().tobytes()

def bigram_ans_decode(data, n):
    buf = np.frombuffer(data, dtype=np.uint32).copy()
    coder = constriction.stream.stack.AnsCoder(buf)
    out = []
    prev = None
    for i in range(n):
        model = uni_model if i == 0 else model_for_prev(prev)
        tok = int(coder.decode(model, 1)[0])
        out.append(tok)
        prev = tok
    return out

# ── ratio + roundtrip check, small sample (per-symbol python loop is slow) ───
print("Loading WikiText-103 test split...")
ds_test = load_dataset("wikitext", "wikitext-103-v1", split="test", trust_remote_code=True)
articles = [t.strip() for t in ds_test["text"] if 50 <= len(t.strip().split()) <= 150][:30]

print(f"\nRatio + roundtrip check on {len(articles)} short articles:")
total_raw = total_bigram = 0
t0 = time.time()
for text in articles:
    raw = text.encode("utf-8")
    ids = enc.encode(text)
    compressed = bigram_ans_encode(ids)
    decoded = bigram_ans_decode(compressed, len(ids))
    assert decoded == ids, "ROUNDTRIP FAILED"
    total_raw += len(raw)
    total_bigram += len(compressed)
print(f"  roundtrip OK, {time.time()-t0:.1f}s total")
print(f"  ratio: {total_raw/total_bigram:.3f}x  (analytic estimate from expA: 4.35x)")
print(f"  distinct prev-token models cached so far: {len(model_cache):,}")

# ── latency: a FRESH process only has the model cache warmed by whatever
# docs it has already served. Measure both a cold single document (empty
# cache, worst case for a brand new service) and steady-state (cache
# already warm from the 30 articles above, realistic for a long-running
# service reusing one shared bigram table across many documents). ────────────
sample_text = articles[5]
sample_ids = enc.encode(sample_text)
n_unique_prev = len(set(sample_ids[:-1]))
print(f"\nLatency on one {len(sample_ids)}-token article ({n_unique_prev} distinct prev-tokens), 5 runs, model cache memoized per prev-token:")

model_cache.clear()
cold_enc_times = []
for _ in range(5):
    model_cache.clear()  # force a fresh cold cache each run to measure true cold cost
    t0 = time.perf_counter()
    compressed = bigram_ans_encode(sample_ids)
    cold_enc_times.append((time.perf_counter()-t0)*1e6)
print(f"  cold-cache encode: {sorted(cold_enc_times)[2]:.0f}us median  ({sorted(cold_enc_times)[2]/len(sample_ids):.2f}us/token)")

# warm the cache once, then measure steady-state (cache already has every
# distinct prev-token this article needs)
bigram_ans_encode(sample_ids)
enc_times = []
for _ in range(5):
    t0 = time.perf_counter()
    compressed = bigram_ans_encode(sample_ids)
    enc_times.append((time.perf_counter()-t0)*1e6)
dec_times = []
for _ in range(5):
    t0 = time.perf_counter()
    bigram_ans_decode(compressed, len(sample_ids))
    dec_times.append((time.perf_counter()-t0)*1e6)
print(f"  warm-cache encode: {sorted(enc_times)[2]:.0f}us median  ({sorted(enc_times)[2]/len(sample_ids):.2f}us/token)")
print(f"  warm-cache decode: {sorted(dec_times)[2]:.0f}us median  ({sorted(dec_times)[2]/len(sample_ids):.2f}us/token)")
print(f"  (for reference: unigram ANS via constriction, vectorized: ~9-17us TOTAL overhead per ~150-token doc)")
