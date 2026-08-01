"""
Kalcher 2026 baseline ("Frequency-Ordered Tokenization for Better Text
Compression", arXiv:2602.22958) benchmarked head-to-head against this repo's
existing token-native methods, on the same corpora / 512-token chunks /
train-test split / bootstrap-CI / median-of-30-reps latency methodology as
`03_latency/bench_agent_mode_v2.py` and `04_frequency_remap/bench_freqremap.py`.

Kalcher's pipeline:
  1. BPE-tokenize (already done -- we use the repo's r50k token-id arrays).
  2. Frequency-reorder the vocab so the most frequent token gets the smallest
     ID. This is the SAME rank table this repo's `+freq` already builds
     (argsort descending frequency on the TRAIN split), so we reuse it.
  3. LEB128 varint encode the remapped IDs (7 value bits + 1 continuation bit
     per byte). Implemented here (see leb128_encode/decode), NOT streamvbyte.
  4. Feed the varint byte stream to a general-purpose compressor. Kalcher's
     Table I tested zlib-9 / LZMA / zstd-22 / bz2 / PPMd; we report the two
     strong ones he highlights: LZMA and zstd-22.

So Kalcher = freq-remap -> LEB128 -> {LZMA, zstd-22}. Contrast with this repo's
`+freq` = freq-remap -> streamvbyte (no general compressor). The comparison is
meant to show: Kalcher buys extra ratio with a general compressor but pays a
much slower decode; `+freq` keeps most of the ratio at ~streamvbyte decode
speed.

Methods reported per domain (r50k token IDs):
  raw            : r50k uint16 packing, no compression
  +freq          : freq-remap -> streamvbyte           (repo baseline)
  +ANS           : static shared unigram ANS            (repo baseline)
  Kalcher(LZMA)  : freq-remap -> LEB128 -> LZMA(9|EXTREME)
  Kalcher(zstd)  : freq-remap -> LEB128 -> zstd-22

Metrics per method: compression ratio vs UTF-8 (median + bootstrap CI over 40
512-token test chunks), plus "agent-facing" decode latency and encode latency
(median-of-30-reps per chunk, then bootstrap over chunks), in microseconds.
Decode is the agent-ready cost: it stops at token IDs (no detokenize), exactly
as the repo's other decode numbers do.
"""
import os
import time
import json
import lzma
import numpy as np
import tiktoken
import zstandard as zstd
import pyfastpfor
import constriction

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
DOMAINS = ["prose", "code", "hindi"]
CHUNK_SIZE = 512
N_CHUNKS = 40
REPS = 30
VOCAB = 50257  # r50k
ANS_TRAIN_TOKENS = 400 * CHUNK_SIZE  # match 01_storage_efficiency/bench_full_results_table.py
N_BOOTSTRAP = 2000
SEED = 3344

r50k = tiktoken.get_encoding("r50k_base")
svb_codec = pyfastpfor.getCodec("streamvbyte")
zstd_c22 = zstd.ZstdCompressor(level=22)
zstd_d = zstd.ZstdDecompressor()
LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]


def bootstrap_ci(values, rng, n_boot=N_BOOTSTRAP, alpha=0.10):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(np.median(values)), (float(values[0]), float(values[0]))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_medians = np.median(values[idx], axis=1)
    lo, hi = np.percentile(boot_medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(values)), (float(lo), float(hi))


def make_chunks(test_arr, chunk_size, n_chunks, rng):
    max_chunks = len(test_arr) // chunk_size
    n = min(n_chunks, max_chunks)
    starts = rng.choice(max_chunks, size=n, replace=False) * chunk_size
    return [test_arr[s : s + chunk_size] for s in starts]


def timed_reps(fn, reps=REPS):
    """Median of `reps` back-to-back calls, in microseconds -- robust per-chunk
    sample instead of a single noise-prone perf_counter() pair (matches
    bench_agent_mode_v2.py)."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e6)
    return float(np.median(ts))


# ── LEB128 varint (Protocol-Buffers style), vectorized in numpy ──────────────
# 7 value bits + 1 continuation bit per byte. Remapped r50k ranks are 0..50256,
# so every value fits in at most 3 bytes (1B: 0-127, 2B: 128-16383,
# 3B: 16384-2097151). Vectorized so we don't strawman Kalcher with a slow
# pure-Python byte loop -- the honest cost is the general compressor, not this.
def leb128_encode(values):
    v = np.asarray(values, dtype=np.uint32)
    assert v.max(initial=0) < (1 << 21), "value exceeds 3-byte LEB128 range"
    nbytes = np.ones(len(v), dtype=np.int64) + (v >= 128) + (v >= 16384)
    ends = np.cumsum(nbytes)
    total = int(ends[-1]) if len(v) else 0
    starts = ends - nbytes
    out = np.zeros(total, dtype=np.uint8)
    b0 = (v & 0x7F).astype(np.uint8) | np.where(nbytes > 1, 0x80, 0).astype(np.uint8)
    out[starts] = b0
    m2 = nbytes > 1
    b1 = ((v[m2] >> 7) & 0x7F).astype(np.uint8) | np.where(
        nbytes[m2] > 2, 0x80, 0
    ).astype(np.uint8)
    out[starts[m2] + 1] = b1
    m3 = nbytes > 2
    b2 = ((v[m3] >> 14) & 0x7F).astype(np.uint8)
    out[starts[m3] + 2] = b2
    return out.tobytes()


def leb128_decode(buf):
    b = np.frombuffer(buf, dtype=np.uint8)
    if len(b) == 0:
        return np.zeros(0, dtype=np.uint32)
    is_last = (b & 0x80) == 0
    # group id for each byte = number of completed values strictly before it
    grp = np.empty(len(b), dtype=np.int64)
    grp[0] = 0
    np.cumsum(is_last[:-1], out=grp[1:])
    n = int(is_last.sum())
    # position of each byte within its group (0,1,2)
    first_idx = np.zeros(n, dtype=np.int64)
    # earliest byte index of each group: since groups are contiguous & ordered,
    # the group start is where grp increments.
    starts = np.searchsorted(grp, np.arange(n))
    k = np.arange(len(b), dtype=np.int64) - starts[grp]
    vals = np.zeros(n, dtype=np.uint64)
    contrib = ((b & 0x7F).astype(np.uint64)) << (7 * k).astype(np.uint64)
    np.add.at(vals, grp, contrib)
    return vals.astype(np.uint32)


# ── streamvbyte (repo +freq container) ───────────────────────────────────────
def svb_encode_arr(arr):
    out = np.zeros(len(arr) * 2 + 1024, dtype=np.uint32)
    n_out = svb_codec.encodeArray(arr.astype(np.uint32), len(arr), out, len(out))
    return out[:n_out].tobytes()


def svb_decode_arr(payload, n):
    packed = np.frombuffer(payload, dtype=np.uint32)
    out = np.zeros(n + 1024, dtype=np.uint32)
    svb_codec.decodeArray(packed, len(packed), out, n)
    return out[:n]


def main():
    rng = np.random.default_rng(SEED)
    results = {}  # (domain, method, metric) -> ci tuple / value
    for domain in DOMAINS:
        print(f"=== {domain} ===", flush=True)
        train = np.load(os.path.join(CORPUS_DIR, f"{domain}_train.npy")).astype(np.int64)
        test = np.load(os.path.join(CORPUS_DIR, f"{domain}_test.npy")).astype(np.int64)

        # +freq / Kalcher rank table: full train split, descending frequency
        counts = np.bincount(train, minlength=VOCAB)
        order = np.argsort(-counts)  # token id, most -> least frequent
        rank_of = np.empty(VOCAB, dtype=np.uint32)
        rank_of[order] = np.arange(VOCAB, dtype=np.uint32)
        token_of_rank = order.astype(np.int64)

        # static ANS model: first 400*512 train tokens (matches repo +ANS)
        ans_counts = np.ones(VOCAB, dtype=np.int64)
        ans_counts += np.bincount(train[:ANS_TRAIN_TOKENS], minlength=VOCAB)
        ans_probs = ans_counts.astype(np.float64) / ans_counts.sum()
        ans_model = constriction.stream.model.Categorical(ans_probs, perfect=False)

        chunks = make_chunks(test, CHUNK_SIZE, N_CHUNKS, rng)
        texts = [r50k.decode(c.tolist()) for c in chunks]
        raw_lens = [len(t.encode("utf-8")) for t in texts]

        # per-method accumulators
        acc = {
            m: {"ratio": [], "enc": [], "dec": []}
            for m in ["raw", "+freq", "+ANS", "Kalcher(LZMA)", "Kalcher(zstd)"]
        }

        for chunk, raw_len in zip(chunks, raw_lens):
            ids = chunk.astype(np.int64)
            n = len(ids)
            remapped = rank_of[ids]  # frequency ranks (uint32)

            # ---- raw: r50k uint16 packing ----
            packed = ids.astype(np.uint16).tobytes()
            acc["raw"]["ratio"].append(raw_len / len(packed))
            acc["raw"]["enc"].append(timed_reps(lambda i=ids: i.astype(np.uint16).tobytes()))
            acc["raw"]["dec"].append(timed_reps(lambda p=packed: np.frombuffer(p, dtype=np.uint16)))

            # ---- +freq: freq-remap -> streamvbyte ----
            svb_payload = svb_encode_arr(remapped)
            assert np.array_equal(svb_decode_arr(svb_payload, n), remapped)
            acc["+freq"]["ratio"].append(raw_len / len(svb_payload))
            acc["+freq"]["enc"].append(
                timed_reps(lambda i=ids: svb_encode_arr(rank_of[i]))
            )

            def freq_decode(p=svb_payload, n=n):
                dr = svb_decode_arr(p, n)
                return token_of_rank[dr]

            acc["+freq"]["dec"].append(timed_reps(freq_decode))

            # ---- +ANS ----
            ids32 = ids.astype(np.int32)

            def ans_encode_once(ids32=ids32):
                c = constriction.stream.stack.AnsCoder()
                c.encode_reverse(ids32, ans_model)
                return c.get_compressed().tobytes()

            ans_payload = ans_encode_once()
            ans_arr = np.frombuffer(ans_payload, dtype=np.uint32).copy()

            def ans_decode_once(a=ans_arr, n=n):
                c2 = constriction.stream.stack.AnsCoder(a.copy())
                return c2.decode(ans_model, n)

            assert np.array_equal(np.asarray(ans_decode_once()), ids32)
            acc["+ANS"]["ratio"].append(raw_len / len(ans_payload))
            acc["+ANS"]["enc"].append(timed_reps(ans_encode_once))
            acc["+ANS"]["dec"].append(timed_reps(ans_decode_once))

            # ---- Kalcher: freq-remap -> LEB128 -> {LZMA, zstd-22} ----
            varint = leb128_encode(remapped)
            assert np.array_equal(leb128_decode(varint), remapped), "LEB128 round-trip"

            lzma_payload = lzma.compress(varint, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
            zstd_payload = zstd_c22.compress(varint)

            # ratio
            acc["Kalcher(LZMA)"]["ratio"].append(raw_len / len(lzma_payload))
            acc["Kalcher(zstd)"]["ratio"].append(raw_len / len(zstd_payload))

            # encode: rank-remap + LEB128 + compress (full text-to-storage minus
            # tokenize, matching how +freq/+ANS agent-encode is timed)
            def kalcher_lzma_encode(i=ids):
                v = leb128_encode(rank_of[i])
                return lzma.compress(v, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)

            def kalcher_zstd_encode(i=ids):
                v = leb128_encode(rank_of[i])
                return zstd_c22.compress(v)

            acc["Kalcher(LZMA)"]["enc"].append(timed_reps(kalcher_lzma_encode))
            acc["Kalcher(zstd)"]["enc"].append(timed_reps(kalcher_zstd_encode))

            # decode: decompress + LEB128 decode + un-permute -> token IDs
            def kalcher_lzma_decode(p=lzma_payload):
                v = lzma.decompress(p, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
                ranks = leb128_decode(v)
                return token_of_rank[ranks]

            def kalcher_zstd_decode(p=zstd_payload, n=n):
                v = zstd_d.decompress(p)
                ranks = leb128_decode(v)
                return token_of_rank[ranks]

            assert np.array_equal(kalcher_lzma_decode(), ids)
            assert np.array_equal(kalcher_zstd_decode(), ids)
            acc["Kalcher(LZMA)"]["dec"].append(timed_reps(kalcher_lzma_decode))
            acc["Kalcher(zstd)"]["dec"].append(timed_reps(kalcher_zstd_decode))

        for m, d in acc.items():
            results[(domain, m, "ratio")] = bootstrap_ci(d["ratio"], rng)
            results[(domain, m, "encode")] = bootstrap_ci(d["enc"], rng)
            results[(domain, m, "decode")] = bootstrap_ci(d["dec"], rng)
        # quick console check
        for m in acc:
            r = results[(domain, m, "ratio")]
            dc = results[(domain, m, "decode")]
            print(f"  {m:<14} ratio={r[0]:.2f}x  decode={dc[0]:.1f}us", flush=True)

    # ── print tables ──
    methods = ["raw", "+freq", "+ANS", "Kalcher(LZMA)", "Kalcher(zstd)"]
    print(f"\n{'='*90}")
    print("COMPRESSION RATIO vs UTF-8 (median [90% bootstrap CI]), 512-token chunks")
    print(f"{'='*90}")
    print(f"  {'method':<16}" + "".join(f"{d:>22}" for d in DOMAINS))
    for m in methods:
        row = f"  {m:<16}"
        for d in DOMAINS:
            v = results[(d, m, "ratio")]
            row += f"{v[0]:.2f}x[{v[1][0]:.2f},{v[1][1]:.2f}]".rjust(22)
        print(row)

    print(f"\n{'='*90}")
    print("DECODE latency to token IDs (median us [90% CI]), 512-token chunks, median-of-30-reps")
    print(f"{'='*90}")
    print(f"  {'method':<16}" + "".join(f"{d:>22}" for d in DOMAINS))
    for m in methods:
        row = f"  {m:<16}"
        for d in DOMAINS:
            v = results[(d, m, "decode")]
            row += f"{v[0]:.1f}[{v[1][0]:.1f},{v[1][1]:.1f}]".rjust(22)
        print(row)

    print(f"\n{'='*90}")
    print("ENCODE latency from token IDs (median us [90% CI]), 512-token chunks")
    print(f"{'='*90}")
    print(f"  {'method':<16}" + "".join(f"{d:>22}" for d in DOMAINS))
    for m in methods:
        row = f"  {m:<16}"
        for d in DOMAINS:
            v = results[(d, m, "encode")]
            row += f"{v[0]:.1f}[{v[1][0]:.1f},{v[1][1]:.1f}]".rjust(22)
        print(row)

    out = {
        "config": {
            "tokenizer": "r50k",
            "chunk_size": CHUNK_SIZE,
            "n_chunks": N_CHUNKS,
            "reps": REPS,
            "seed": SEED,
            "ans_train_tokens": ANS_TRAIN_TOKENS,
            "lzma": "FORMAT_RAW LZMA2 preset=9|EXTREME",
            "zstd": "level 22",
            "note": "decode/encode are agent-facing (token IDs <-> storage), no (de)tokenize; ratio excludes any length header",
        },
        "results": {f"{d}|{m}|{k}": v for (d, m, k), v in results.items()},
    }
    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results.json")


if __name__ == "__main__":
    main()
