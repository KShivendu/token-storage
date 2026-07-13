"""
o200k (GPT-4o tokenizer): compression ratio + full-pipeline latency,
matching the exact methodology already used for r50k/cl100k in the blog
(bench_latency_static.py + bench_cl100k_packing.py combined into one pass
so the numbers are directly comparable).
"""
import os, time
import numpy as np
import tiktoken
import constriction
from datasets import load_dataset

CACHE_DIR = os.path.dirname(__file__)
N_ARTICLES = 200
N_RUNS = 10

enc = tiktoken.get_encoding("o200k_base")
probs = np.load(os.path.join(CACHE_DIR, "static_probs_o200k.npy"))
model = constriction.stream.model.Categorical(probs, perfect=False)

def median_p99(times_us):
    t = sorted(times_us)
    return t[len(t) // 2], t[int(len(t) * 0.99)]

def pack_24(ids):
    arr = np.array(ids, dtype=np.uint32)
    out = np.zeros((len(arr), 3), dtype=np.uint8)
    out[:, 0] = (arr >> 16) & 0xFF
    out[:, 1] = (arr >> 8) & 0xFF
    out[:, 2] = arr & 0xFF
    return out.tobytes()

def unpack_24(data):
    arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
    return (arr[:, 0].astype(np.uint32) << 16) | (arr[:, 1].astype(np.uint32) << 8) | arr[:, 2]

print(f"Loading {N_ARTICLES} WikiText-103 test articles...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test")
articles = [t.strip() for t in ds["text"] if 50 <= len(t.strip().split()) <= 500][:N_ARTICLES]
print(f"  Got {len(articles)} articles")

# ── ratio ─────────────────────────────────────────────────────────────────────

total_raw = total_32 = total_24 = total_ans = 0
for text in articles:
    raw = text.encode("utf-8")
    ids = enc.encode(text)
    total_raw += len(raw)
    total_32 += len(ids) * 4
    total_24 += len(pack_24(ids))
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(np.array(ids, dtype=np.int32), model)
    total_ans += len(coder.get_compressed().tobytes())

print("\n" + "=" * 60)
print("  o200k compression ratio, 200 WikiText-103 articles")
print("=" * 60)
print(f"  uint32 raw:  {total_raw/total_32:.3f}x")
print(f"  3-byte raw:  {total_raw/total_24:.3f}x")
print(f"  + static ANS: {total_raw/total_ans:.3f}x")

# ── latency: tokenizer only, +3-byte (full pipeline), +ANS (full pipeline) ───

tok_enc, tok_dec = [], []
b3_enc, b3_dec = [], []
ans_enc, ans_dec = [], []

for text in articles:
    ids_ref = enc.encode(text)
    packed_ref = pack_24(ids_ref)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(np.array(ids_ref, dtype=np.int32), model)
    compressed_ref = coder.get_compressed().tobytes()
    n = len(ids_ref)

    for _ in range(N_RUNS):
        # tokenizer only
        t0 = time.perf_counter()
        ids = enc.encode(text)
        tok_enc.append((time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter()
        _ = enc.decode(ids)
        tok_dec.append((time.perf_counter() - t0) * 1e6)

        # tokenize + 3-byte pack (full pipeline)
        t0 = time.perf_counter()
        ids2 = enc.encode(text)
        packed = pack_24(ids2)
        b3_enc.append((time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter()
        unpacked = unpack_24(packed_ref)
        _ = enc.decode(unpacked.tolist())
        b3_dec.append((time.perf_counter() - t0) * 1e6)

        # tokenize + ANS (full pipeline)
        t0 = time.perf_counter()
        ids3 = enc.encode(text)
        c2 = constriction.stream.stack.AnsCoder()
        c2.encode_reverse(np.array(ids3, dtype=np.int32), model)
        _ = c2.get_compressed().tobytes()
        ans_enc.append((time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter()
        buf = np.frombuffer(compressed_ref, dtype=np.uint32).copy()
        dec_ids = constriction.stream.stack.AnsCoder(buf).decode(model, n).tolist()
        _ = enc.decode(dec_ids)
        ans_dec.append((time.perf_counter() - t0) * 1e6)

print("\n" + "=" * 70)
print("  o200k latency, median / p99, 200 articles x 10 runs")
print("=" * 70)
for label, e, d in [
    ("tokenizer only", tok_enc, tok_dec),
    ("+ 3-byte", b3_enc, b3_dec),
    ("+ ANS", ans_enc, ans_dec),
]:
    em, ep = median_p99(e)
    dm, dp = median_p99(d)
    print(f"  {label:<16} enc {em:>6.0f}us / {ep:>6.0f}us   dec {dm:>6.0f}us / {dp:>6.0f}us")
print("=" * 70)
