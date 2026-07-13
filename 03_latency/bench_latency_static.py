"""
Latency benchmark: r50k and cl100k full encode/decode pipeline.
Uses static corpus ANS model. Runs over 200 WikiText-103 test articles,
measures median + p99 across all (article, run) pairs.
"""

import os, time
import tiktoken
import numpy as np
import constriction
from datasets import load_dataset

CACHE_DIR = os.path.dirname(__file__)
N_ARTICLES = 200
N_RUNS = 10  # passes per article → 200×10 = 2000 samples

TOKENIZERS = {
    "r50k":   ("r50k_base",   50_257),
    "cl100k": ("cl100k_base", 100_277),
}

# ── load static models ────────────────────────────────────────────────────────

encs, models = {}, {}
for name, (enc_name, vocab_size) in TOKENIZERS.items():
    probs = np.load(os.path.join(CACHE_DIR, f"static_probs_{name}.npy"))
    encs[name]   = tiktoken.get_encoding(enc_name)
    models[name] = constriction.stream.model.Categorical(probs, perfect=False)
    print(f"  {name}: loaded static model ({vocab_size:,} tokens)")

# ── load articles ─────────────────────────────────────────────────────────────

print(f"\nLoading {N_ARTICLES} WikiText-103 test articles...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test")
articles = [t.strip() for t in ds["text"] if 50 <= len(t.strip().split()) <= 500][:N_ARTICLES]
print(f"  Got {len(articles)} articles")

# ── timing harness ────────────────────────────────────────────────────────────

def median_p99(times_us):
    t = sorted(times_us)
    return t[len(t) // 2], t[int(len(t) * 0.99)]

# ── measure ───────────────────────────────────────────────────────────────────

print(f"\nBenchmarking ({N_ARTICLES} articles × {N_RUNS} runs each)...")

rows = []

for name in ("r50k", "cl100k"):
    enc   = encs[name]
    model = models[name]

    enc_times, dec_times = [], []

    for text in articles:
        # pre-encode once so decode has something to work with
        ids = enc.encode(text)
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(np.array(ids, dtype=np.int32), model)
        compressed = coder.get_compressed().tobytes()
        n = len(ids)

        for _ in range(N_RUNS):
            # encode
            t0 = time.perf_counter()
            ids2 = enc.encode(text)
            c2 = constriction.stream.stack.AnsCoder()
            c2.encode_reverse(np.array(ids2, dtype=np.int32), model)
            _ = c2.get_compressed().tobytes()
            enc_times.append((time.perf_counter() - t0) * 1e6)

            # decode
            t0 = time.perf_counter()
            buf = np.frombuffer(compressed, dtype=np.uint32).copy()
            dec_ids = constriction.stream.stack.AnsCoder(buf).decode(model, n).tolist()
            _ = enc.decode(dec_ids)
            dec_times.append((time.perf_counter() - t0) * 1e6)

    e_med, e_p99 = median_p99(enc_times)
    d_med, d_p99 = median_p99(dec_times)
    rows.append((name, e_med, e_p99, d_med, d_p99))
    print(f"  {name}: encode {e_med:.0f}µs / {e_p99:.0f}µs  |  decode {d_med:.0f}µs / {d_p99:.0f}µs")

# ── table ─────────────────────────────────────────────────────────────────────

print(f"\n{'='*72}")
print(f"  Pipeline latency — {N_ARTICLES} WikiText articles, static ANS model")
print(f"{'='*72}")
print(f"  {'Method':<12}  {'enc median':>10}  {'enc p99':>8}  {'dec median':>10}  {'dec p99':>8}")
print(f"  {'-'*60}")
for name, em, ep, dm, dp in rows:
    print(f"  {name:<12}  {em:>9.0f}µs  {ep:>7.0f}µs  {dm:>9.0f}µs  {dp:>7.0f}µs")
print(f"{'='*72}")
