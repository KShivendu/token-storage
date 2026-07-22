"""
Splits the +freq method's encode/decode into its real two components, since
the earlier freqEnc/freqDec numbers only timed the streamvbyte codec calls
(the numpy rank-remap/rank-lookup step was done outside the timed region in
the original scripts). Robust methodology: median of 30 reps per chunk.

  - rank_remap_only:    ids -> remapped ids (rank_of[ids], numpy fancy index)
  - streamvbyte_enc:    remapped ids -> packed bytes
  - streamvbyte_dec:    packed bytes -> remapped ids
  - rank_lookup_only:   remapped ids -> ids (token_of_rank[remapped], numpy fancy index)
"""
import os
import time
import numpy as np
import tiktoken
import pyfastpfor

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
DOMAINS = ["prose", "code", "hindi"]
CHUNK_SIZE = 512
N_CHUNKS = 40
REPS = 30
TOKENIZERS = {"r50k": ("r50k_base", 50257), "cl100k": ("cl100k_base", 100277), "o200k": ("o200k_base", 200019)}
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(3344)

r50k = tiktoken.get_encoding("r50k_base")
svb_codec = pyfastpfor.getCodec("streamvbyte")


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
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e6)
    return float(np.median(ts))


def svb_encode_arr(arr):
    out = np.zeros(len(arr) * 2 + 1024, dtype=np.uint32)
    n_out = svb_codec.encodeArray(arr, len(arr), out, len(out))
    return out[:n_out].tobytes()


def svb_decode_arr(payload, n):
    packed = np.frombuffer(payload, dtype=np.uint32)
    out = np.zeros(n + 1024, dtype=np.uint32)
    svb_codec.decodeArray(packed, len(packed), out, n)
    return out[:n]


results = {}
for domain in DOMAINS:
    print(f"=== {domain} ===", flush=True)
    train_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_train.npy")).astype(np.int64)
    test_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_test.npy")).astype(np.int64)
    train_text = r50k.decode(train_r50k.tolist())

    chunks_r50k = make_chunks(test_r50k, CHUNK_SIZE, N_CHUNKS)
    texts = [r50k.decode(c.tolist()) for c in chunks_r50k]

    for tok_key, (enc_name, vocab_size) in TOKENIZERS.items():
        enc = tiktoken.get_encoding(enc_name)
        train_ids = enc.encode(train_text, disallowed_special=())
        counts = np.zeros(vocab_size, dtype=np.int64)
        vals, cnts = np.unique(train_ids, return_counts=True)
        counts[vals] = cnts
        order = np.argsort(-counts)
        rank_of = np.empty(vocab_size, dtype=np.uint32)
        rank_of[order] = np.arange(vocab_size, dtype=np.uint32)
        token_of_rank = order.astype(np.int64)

        remap_t, svbenc_t, svbdec_t, lookup_t = [], [], [], []
        for text in texts:
            ids = enc.encode(text, disallowed_special=())
            ids_arr = np.array(ids, dtype=np.int64)

            remap_t.append(timed_reps(lambda a=ids_arr: rank_of[a]))
            remapped = rank_of[ids_arr]

            svbenc_t.append(timed_reps(lambda r=remapped: svb_encode_arr(r)))
            payload = svb_encode_arr(remapped)

            svbdec_t.append(timed_reps(lambda p=payload, n=len(ids): svb_decode_arr(p, n)))
            decoded_remapped = svb_decode_arr(payload, len(ids))

            lookup_t.append(timed_reps(lambda d=decoded_remapped: token_of_rank[d]))

            assert list(token_of_rank[decoded_remapped]) == ids

        results[(domain, tok_key, "rank_remap_only")] = bootstrap_ci(remap_t)
        results[(domain, tok_key, "streamvbyte_enc")] = bootstrap_ci(svbenc_t)
        results[(domain, tok_key, "streamvbyte_dec")] = bootstrap_ci(svbdec_t)
        results[(domain, tok_key, "rank_lookup_only")] = bootstrap_ci(lookup_t)
        print(f"  {tok_key} remap/svb-enc/svb-dec/lookup done", flush=True)

print(f"\n{'=' * 100}")
print("  +freq split (robust, median-of-30-reps per chunk), chunk_size=512, median us")
print(f"{'=' * 100}")
for domain in DOMAINS:
    print(f"\n-- {domain} --")
    for tok_key in TOKENIZERS:
        r = results[(domain, tok_key, "rank_remap_only")]
        se = results[(domain, tok_key, "streamvbyte_enc")]
        sd = results[(domain, tok_key, "streamvbyte_dec")]
        l = results[(domain, tok_key, "rank_lookup_only")]
        print(
            f"  {tok_key:<8} rank_remap={r[0]:>7.3f}us  streamvbyte_enc={se[0]:>7.3f}us  "
            f"streamvbyte_dec={sd[0]:>7.3f}us  rank_lookup={l[0]:>7.3f}us"
        )

import json
with open(os.path.join(os.path.dirname(__file__), "freq_split_results.json"), "w") as f:
    json.dump({f"{d}|{t}|{m}": v for (d, t, m), v in results.items()}, f, indent=2)
print("\nSaved to freq_split_results.json")
