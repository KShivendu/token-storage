"""
v2 of bench_agent_mode.py: fixes a real measurement bug the user caught (single
perf_counter() shots on sub-microsecond/few-microsecond ops are dominated by
GC/scheduler noise, not the operation itself -- e.g. r50k raw unpack measured
slower than LZ4 decompress, which made no sense for a zero-copy reinterpret vs
real decompression). Fix: for every FAST op (decompress_only, raw_unpack_only,
ans_decode_only), time REPS=30 back-to-back calls per chunk and take the median
of those reps as that chunk's cost, then bootstrap across the 40 chunks as
before. tokenize_only stays single-shot (100s of us, not noise-sensitive at
this scale) for consistency with the rest of the post's methodology.
"""
import os
import time
import numpy as np
import tiktoken
import lz4.frame as lz4f
import gzip
import zstandard as zstd
import brotli
import constriction

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
DOMAINS = ["prose", "code", "hindi"]
CHUNK_SIZE = 512
N_CHUNKS = 40
REPS = 30
TOKENIZERS = {"r50k": ("r50k_base", 50257), "cl100k": ("cl100k_base", 100277), "o200k": ("o200k_base", 200019)}
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(3344)

r50k = tiktoken.get_encoding("r50k_base")
zstd_c19 = zstd.ZstdCompressor(level=19)
zstd_d = zstd.ZstdDecompressor()


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=0.10):
    values = np.asarray(values, dtype=np.float64)
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    boot_medians = np.median(values[idx], axis=1)
    lo, hi = np.percentile(boot_medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(values)), (float(lo), float(hi))


def make_chunks(test_arr, chunk_size, n_chunks):
    max_chunks = len(test_arr) // chunk_size
    n = min(n_chunks, max_chunks)
    starts = RNG.choice(max_chunks, size=n, replace=False) * chunk_size
    return [test_arr[s: s + chunk_size] for s in starts]


def timed_reps(fn, reps=REPS):
    """Median of `reps` back-to-back calls, in microseconds -- one robust
    per-chunk sample instead of a single noise-prone perf_counter() pair."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e6)
    return float(np.median(ts))


def pack3(ids):
    n = len(ids)
    out = np.zeros(n * 3, dtype=np.uint8)
    out[0::3] = (ids >> 16) & 0xFF
    out[1::3] = (ids >> 8) & 0xFF
    out[2::3] = ids & 0xFF
    return out.tobytes()


def unpack3(buf, n):
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, 3).astype(np.int64)
    return (arr[:, 0] << 16) | (arr[:, 1] << 8) | arr[:, 2]


results = {}
for domain in DOMAINS:
    print(f"=== {domain} ===", flush=True)
    train_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_train.npy")).astype(np.int64)
    test_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_test.npy")).astype(np.int64)
    train_text = r50k.decode(train_r50k.tolist())

    chunks_r50k = make_chunks(test_r50k, CHUNK_SIZE, N_CHUNKS)
    texts = [r50k.decode(c.tolist()) for c in chunks_r50k]
    raw_byte_lists = [t.encode("utf-8") for t in texts]

    # ---- byte codecs: decompress-only (robust) ----
    codecs = {
        "LZ4": (lambda b: lz4f.compress(b), lambda c: lz4f.decompress(c)),
        "gzip-9": (lambda b: gzip.compress(b, 9), lambda c: gzip.decompress(c)),
        "zstd-19": (lambda b: zstd_c19.compress(b), lambda c: zstd_d.decompress(c)),
        "brotli-q11": (lambda b: brotli.compress(b, quality=11), lambda c: brotli.decompress(c)),
    }
    # Compress ONCE per chunk, then repeatedly decompress that same payload
    # (that's what "decompress_only" should time, not recompression cost).
    for name, (comp, decomp) in codecs.items():
        dec_t = []
        for raw in raw_byte_lists:
            c = comp(raw)
            dec_t.append(timed_reps(lambda c=c: decomp(c)))
        results[(domain, name, "decompress_only")] = bootstrap_ci(dec_t)
        print(f"  {name} decompress_only done", flush=True)

    # zstd --train: 112KB dict trained on this domain's train split
    train_chunks_bytes = [
        r50k.decode(train_r50k[i: i + CHUNK_SIZE].tolist()).encode("utf-8")
        for i in range(0, min(len(train_r50k), CHUNK_SIZE * 400), CHUNK_SIZE)
    ]
    zdict = zstd.ZstdCompressionDict(zstd.train_dictionary(112 * 1024, train_chunks_bytes).as_bytes())
    zc_dict = zstd.ZstdCompressor(level=19, dict_data=zdict)
    zd_dict = zstd.ZstdDecompressor(dict_data=zdict)
    dec_t = []
    for raw in raw_byte_lists:
        c = zc_dict.compress(raw)
        dec_t.append(timed_reps(lambda c=c: zd_dict.decompress(c)))
    results[(domain, "zstd --train", "decompress_only")] = bootstrap_ci(dec_t)
    print("  zstd --train decompress_only done", flush=True)

    # ---- per-tokenizer: tokenize-only (single-shot, large enough), raw
    # unpack-only (robust), ANS decode-only (robust) ----
    for tok_key, (enc_name, vocab_size) in TOKENIZERS.items():
        enc = tiktoken.get_encoding(enc_name)
        train_ids = enc.encode(train_text, disallowed_special=())
        counts = np.ones(vocab_size, dtype=np.int64)
        counts += np.bincount(train_ids, minlength=vocab_size)
        probs = counts.astype(np.float64) / counts.sum()
        model = constriction.stream.model.Categorical(probs, perfect=False)

        tok_t, unpack_t, ans_dec_t = [], [], []
        for text in texts:
            ids = enc.encode(text, disallowed_special=())
            ids_arr = np.array(ids, dtype=np.int64)

            t0 = time.perf_counter()
            retok = enc.encode(text, disallowed_special=())
            t1 = time.perf_counter()
            tok_t.append((t1 - t0) * 1e6)

            if tok_key == "r50k":
                packed = ids_arr.astype(np.uint16).tobytes()
                unpack_t.append(timed_reps(lambda p=packed: np.frombuffer(p, dtype=np.uint16)))
            else:
                packed = pack3(ids_arr)
                n = len(ids_arr)
                unpack_t.append(timed_reps(lambda p=packed, n=n: unpack3(p, n)))

            c = constriction.stream.stack.AnsCoder()
            c.encode_reverse(np.array(ids, dtype=np.int32), model)
            payload = np.frombuffer(c.get_compressed().tobytes(), dtype=np.uint32).copy()
            n_ids = len(ids)

            def ans_decode_once(payload=payload, n_ids=n_ids):
                c2 = constriction.stream.stack.AnsCoder(payload.copy())
                return c2.decode(model, n_ids)

            # sanity check once per chunk (not timed)
            assert list(ans_decode_once()) == ids
            ans_dec_t.append(timed_reps(ans_decode_once))

        results[(domain, tok_key, "tokenize_only")] = bootstrap_ci(tok_t)
        results[(domain, tok_key, "raw_unpack_only")] = bootstrap_ci(unpack_t)
        results[(domain, tok_key, "ans_decode_only")] = bootstrap_ci(ans_dec_t)
        print(f"  {tok_key} tokenize/unpack/ans-decode done", flush=True)

print(f"\n{'=' * 100}")
print("  Agent-decode components (robust, median-of-30-reps per chunk), chunk_size=512, median us")
print(f"{'=' * 100}")
for domain in DOMAINS:
    print(f"\n-- {domain} --")
    for name in ["LZ4", "gzip-9", "zstd-19", "brotli-q11", "zstd --train"]:
        v = results[(domain, name, "decompress_only")]
        print(f"  {name:<14} decompress_only={v[0]:.2f}us [{v[1][0]:.2f},{v[1][1]:.2f}]")
    for tok_key in TOKENIZERS:
        t = results[(domain, tok_key, "tokenize_only")]
        u = results[(domain, tok_key, "raw_unpack_only")]
        a = results[(domain, tok_key, "ans_decode_only")]
        print(f"  {tok_key:<8} tokenize_only={t[0]:>8.2f}us  raw_unpack_only={u[0]:>7.3f}us  ans_decode_only={a[0]:>7.3f}us")

import json
with open(os.path.join(os.path.dirname(__file__), "agent_mode_results_v2.json"), "w") as f:
    json.dump({f"{d}|{n}|{m}": v for (d, n, m), v in results.items()}, f, indent=2)
print("\nSaved to agent_mode_results_v2.json")
