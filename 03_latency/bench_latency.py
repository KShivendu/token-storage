"""
Latency benchmark: compress/decompress across all methods on 200 WikiText-103 articles.
Uses static corpus ANS model. Reports median + p99 per method.
"""

import gzip, os, struct, time, zlib
import lz4.frame
import zstandard as zstd
import brotli
import tiktoken
import numpy as np
import constriction
from datasets import load_dataset

CACHE_DIR = os.path.dirname(__file__)
N_ARTICLES = 200
N_RUNS = 10  # passes per article → 2000 samples total

# ── setup ─────────────────────────────────────────────────────────────────────

TOKENIZERS = {
    "r50k":   ("r50k_base",   50_257),
    "cl100k": ("cl100k_base", 100_277),
}
encs, ans_models = {}, {}
for name, (enc_name, vocab_size) in TOKENIZERS.items():
    encs[name] = tiktoken.get_encoding(enc_name)
    probs = np.load(os.path.join(CACHE_DIR, f"static_probs_{name}.npy"))
    ans_models[name] = constriction.stream.model.Categorical(probs, perfect=False)

dict_path = os.path.join(CACHE_DIR, "zstd_dict_112k.bin")
with open(dict_path, "rb") as f:
    zstd_dict_data = f.read()
zdict = zstd.ZstdCompressionDict(zstd_dict_data)
zstd_c        = zstd.ZstdCompressor(level=22)
zstd_dict_c   = zstd.ZstdCompressor(level=22, dict_data=zdict)
zstd_d        = zstd.ZstdDecompressor()
zstd_dict_d   = zstd.ZstdDecompressor(dict_data=zdict)

tok_dict_path = os.path.join(CACHE_DIR, "zstd_dict_token_112k.bin")
with open(tok_dict_path, "rb") as f:
    tok_dict_data = f.read()
tok_zdict        = zstd.ZstdCompressionDict(tok_dict_data)
zstd_tokdict_c   = zstd.ZstdCompressor(level=22, dict_data=tok_zdict)
zstd_tokdict_d   = zstd.ZstdDecompressor(dict_data=tok_zdict)

# naive 32KB zlib preset dict (RFC 1950 zdict, not ZDICT-trained), text + token bytes
ZLIB_DICT_SIZE = 32 * 1024

def build_zlib_preset_dict(samples, cap=ZLIB_DICT_SIZE):
    buf = bytearray()
    for s in samples:
        if len(buf) >= cap:
            break
        buf += s
    return bytes(buf[:cap])

print("Building zlib preset dictionaries (32KB, naive)...")
ds_train = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
train_texts = []
for t in ds_train["text"]:
    t = t.strip()
    if len(t.split()) >= 30:
        train_texts.append(t)
    if len(train_texts) >= 200:  # plenty to fill a 32KB dict, avoid scanning full split
        break

zlib_text_dict = build_zlib_preset_dict([t.encode("utf-8") for t in train_texts])
zlib_tok_dict = build_zlib_preset_dict([
    struct.pack(f">{len(ids)}H", *ids)
    for ids in (encs["r50k"].encode(t) for t in train_texts)
    if ids
])

def zlib_compress(data, zd):
    co = zlib.compressobj(level=9, wbits=15, zdict=zd)
    return co.compress(data) + co.flush()

def zlib_decompress(data, zd):
    do = zlib.decompressobj(wbits=15, zdict=zd)
    return do.decompress(data) + do.flush()

# ── load articles ─────────────────────────────────────────────────────────────

print(f"Loading {N_ARTICLES} WikiText-103 test articles...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test")
articles = [t.strip() for t in ds["text"] if 50 <= len(t.strip().split()) <= 500][:N_ARTICLES]
print(f"  Got {len(articles)} articles\n")

# ── timing harness ────────────────────────────────────────────────────────────

def median_p99(times_us):
    t = sorted(times_us)
    return t[len(t) // 2], t[int(len(t) * 0.99)]

def bench(compress_fn, decompress_fn):
    enc_times, dec_times = [], []
    for text in articles:
        raw = text.encode("utf-8")
        compressed = compress_fn(raw)
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            compress_fn(raw)
            enc_times.append((time.perf_counter() - t0) * 1e6)
            t0 = time.perf_counter()
            decompress_fn(compressed)
            dec_times.append((time.perf_counter() - t0) * 1e6)
    return median_p99(enc_times), median_p99(dec_times)

def bench_tok(tok_name):
    enc = encs[tok_name]
    model = ans_models[tok_name]
    enc_times, dec_times = [], []
    for text in articles:
        ids = enc.encode(text)
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(np.array(ids, dtype=np.int32), model)
        compressed = coder.get_compressed().tobytes()
        n = len(ids)
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            ids2 = enc.encode(text)
            c2 = constriction.stream.stack.AnsCoder()
            c2.encode_reverse(np.array(ids2, dtype=np.int32), model)
            c2.get_compressed().tobytes()
            enc_times.append((time.perf_counter() - t0) * 1e6)
            t0 = time.perf_counter()
            buf = np.frombuffer(compressed, dtype=np.uint32).copy()
            dec_ids = constriction.stream.stack.AnsCoder(buf).decode(model, n).tolist()
            enc.decode(dec_ids)
            dec_times.append((time.perf_counter() - t0) * 1e6)
    return median_p99(enc_times), median_p99(dec_times)

def bench_tok_only(tok_name):
    """Pure tokenizer encode/decode, no ANS, to isolate the tokenizer's share of bench_tok's cost."""
    enc = encs[tok_name]
    enc_times, dec_times = [], []
    for text in articles:
        ids = enc.encode(text)
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            enc.encode(text)
            enc_times.append((time.perf_counter() - t0) * 1e6)
            t0 = time.perf_counter()
            enc.decode(ids)
            dec_times.append((time.perf_counter() - t0) * 1e6)
    return median_p99(enc_times), median_p99(dec_times)

def bench_tok_bytes(tok_name, compress_fn, decompress_fn):
    """Full round trip: tokenize -> pack uint16 -> byte-codec compress/decompress -> unpack -> detokenize."""
    enc = encs[tok_name]
    enc_times, dec_times = [], []
    for text in articles:
        ids = enc.encode(text)
        packed = struct.pack(f">{len(ids)}H", *ids)
        compressed = compress_fn(packed)
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            ids2 = enc.encode(text)
            packed2 = struct.pack(f">{len(ids2)}H", *ids2)
            compress_fn(packed2)
            enc_times.append((time.perf_counter() - t0) * 1e6)
            t0 = time.perf_counter()
            packed_out = decompress_fn(compressed)
            ids_out = struct.unpack(f">{len(packed_out) // 2}H", packed_out)
            enc.decode(list(ids_out))
            dec_times.append((time.perf_counter() - t0) * 1e6)
    return median_p99(enc_times), median_p99(dec_times)

# ── run ───────────────────────────────────────────────────────────────────────

rows = []

def add(label, *args):
    (em, ep), (dm, dp) = args[0](*args[1:]) if len(args) > 1 else args[0]
    rows.append((label, em, ep, dm, dp))
    print(f"  {label}")

print("Benchmarking...")
add("LZ4",           bench,
    lambda r: lz4.frame.compress(r, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX),
    lambda c: lz4.frame.decompress(c))
add("gzip -9",       bench,
    lambda r: gzip.compress(r, compresslevel=9),
    lambda c: gzip.decompress(c))
add("zstd -22",      bench,
    lambda r: zstd_c.compress(r),
    lambda c: zstd_d.decompress(c))
add("zstd dict",     bench,
    lambda r: zstd_dict_c.compress(r),
    lambda c: zstd_dict_d.decompress(c))
add("r50k + ANS",    bench_tok, "r50k")
add("cl100k + ANS",  bench_tok, "cl100k")
add("r50k tokenizer only",    bench_tok_only, "r50k")
add("cl100k tokenizer only",  bench_tok_only, "cl100k")
add("zlib presetdict (text)", bench,
    lambda r: zlib_compress(r, zlib_text_dict),
    lambda c: zlib_decompress(c, zlib_text_dict))
add("r50k uint16 + lz4",       bench_tok_bytes, "r50k",
    lambda p: lz4.frame.compress(p, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX),
    lambda c: lz4.frame.decompress(c))
add("r50k uint16 + gzip",      bench_tok_bytes, "r50k",
    lambda p: gzip.compress(p, compresslevel=9),
    lambda c: gzip.decompress(c))
add("r50k uint16 + zstd tokdict", bench_tok_bytes, "r50k",
    lambda p: zstd_tokdict_c.compress(p),
    lambda c: zstd_tokdict_d.decompress(c))
add("r50k uint16 + zlib presetdict", bench_tok_bytes, "r50k",
    lambda p: zlib_compress(p, zlib_tok_dict),
    lambda c: zlib_decompress(c, zlib_tok_dict))

# ── table ─────────────────────────────────────────────────────────────────────

W = 96
print(f"\n{'='*W}")
print(f"  Latency — {N_ARTICLES} WikiText articles × {N_RUNS} runs  (static ANS model)")
print(f"{'='*W}")
print(f"  {'Method':<36}  {'enc median':>10}  {'enc p99':>8}  {'dec median':>10}  {'dec p99':>8}")
print(f"  {'-'*88}")
for label, em, ep, dm, dp in rows:
    print(f"  {label:<36}  {em:>9.0f}µs  {ep:>7.0f}µs  {dm:>9.0f}µs  {dp:>7.0f}µs")
print(f"{'='*W}")
