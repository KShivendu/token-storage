"""
Compression + latency analysis of cl100k packing widths: uint32 (current),
3-byte/24-bit (byte-aligned), and 17-bit (exact bit-packed). No ANS involved,
this is purely about the raw token-native storage representation.
"""
import time
import numpy as np
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("cl100k_base")
BITS = 17
N_ARTICLES = 200
N_RUNS = 10

def median_p99(times_us):
    t = sorted(times_us)
    return t[len(t) // 2], t[int(len(t) * 0.99)]

# ── packing implementations ──────────────────────────────────────────────────

def pack_uint32(ids):
    return np.array(ids, dtype=np.uint32).tobytes()

def unpack_uint32(data):
    return np.frombuffer(data, dtype=np.uint32)

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

def pack_17(ids):
    acc, acc_bits, buf = 0, 0, bytearray()
    for v in ids:
        acc = (acc << BITS) | int(v)
        acc_bits += BITS
        while acc_bits >= 8:
            acc_bits -= 8
            buf.append((acc >> acc_bits) & 0xFF)
    if acc_bits:
        buf.append((acc << (8 - acc_bits)) & 0xFF)
    return bytes(buf)

def unpack_17(data, n):
    bitsarr = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: n * BITS]
    bitsarr = bitsarr.reshape(n, BITS)
    weights = (1 << np.arange(BITS - 1, -1, -1)).astype(np.uint32)
    return (bitsarr.astype(np.uint32) * weights).sum(axis=1)

# ── load corpus (same methodology as main benchmark) ─────────────────────────

print(f"Loading {N_ARTICLES} WikiText-103 test articles...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test")
articles = [t.strip() for t in ds["text"] if 50 <= len(t.strip().split()) <= 500][:N_ARTICLES]
print(f"  Got {len(articles)} articles\n")

# ── compression ratio, full set ───────────────────────────────────────────────

total_raw = total_32 = total_24 = total_17 = 0
for text in articles:
    raw = text.encode("utf-8")
    ids = enc.encode(text)
    total_raw += len(raw)
    total_32 += len(ids) * 4
    total_24 += len(ids) * 3
    total_17 += len(pack_17(ids))

print("=" * 70)
print("  Compression ratio (cl100k, no ANS), 200 WikiText-103 articles")
print("=" * 70)
print(f"  {'Method':<20} {'Bytes':>12} {'Ratio':>8}")
print(f"  {'-'*44}")
print(f"  {'raw UTF-8':<20} {total_raw:>12,} {1.0:>7.2f}x")
print(f"  {'uint32 (current)':<20} {total_32:>12,} {total_raw/total_32:>7.2f}x")
print(f"  {'3-byte/24-bit':<20} {total_24:>12,} {total_raw/total_24:>7.2f}x")
print(f"  {'17-bit packed':<20} {total_17:>12,} {total_raw/total_17:>7.2f}x")

# ── latency, matching bench_latency.py methodology ───────────────────────────

def bench(pack_fn, unpack_fn, needs_n=False):
    enc_times, dec_times = [], []
    for text in articles:
        ids = enc.encode(text)
        packed = pack_fn(ids)
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            p = pack_fn(ids)
            enc_times.append((time.perf_counter() - t0) * 1e6)
            t0 = time.perf_counter()
            if needs_n:
                unpack_fn(packed, len(ids))
            else:
                unpack_fn(packed)
            dec_times.append((time.perf_counter() - t0) * 1e6)
    return median_p99(enc_times), median_p99(dec_times)

print("\n" + "=" * 70)
print("  Latency (cl100k, no ANS), 200 articles x 10 runs")
print("=" * 70)
print(f"  {'Method':<20} {'enc median':>10} {'enc p99':>9} {'dec median':>10} {'dec p99':>9}")
print(f"  {'-'*62}")

for label, pack_fn, unpack_fn, needs_n in [
    ("uint32", pack_uint32, unpack_uint32, False),
    ("3-byte/24-bit", pack_24, unpack_24, False),
    ("17-bit packed", pack_17, unpack_17, True),
]:
    (em, ep), (dm, dp) = bench(pack_fn, unpack_fn, needs_n)
    print(f"  {label:<20} {em:>9.2f}us {ep:>8.2f}us {dm:>9.2f}us {dp:>8.2f}us")

print("=" * 70)
