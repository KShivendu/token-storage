"""
Experiment #1: does a CORPUS-WIDE trained zstd dictionary let a *fast-decode*
token-native method match/beat Kalcher-LZMA on CODE compression?

Motivation: code has heavy cross-file verbatim repetition (imports, boilerplate,
idioms, repeated identifiers) that per-chunk zeroth-order methods (+freq, +ANS)
cannot see. A dictionary trained over the WHOLE code train split primes zstd's
window with that shared structure -- captured at fast zstd decode speed instead
of LZMA's slow range coder.

Pipeline (token domain, LZ-friendly): each chunk -> cl100k ids -> freq-remap
(most frequent token -> smallest id, full-train rank table) -> LEB128 varint
(contiguous per-token bytes, NOT streamvbyte's split control/data layout which
is LZ-hostile) -> zstd-22 compress WITH the corpus-wide trained dict.

Controlled comparison (isolates the dict's contribution):
    Kalcher-zstd (freq + LEB128 + zstd-22, per-chunk, NO dict)
        vs
    same pipeline + corpus-wide trained dict         <- the delta = dict effect

Reference rows (same held-out test chunks): +freq, +ANS, Kalcher-LZMA,
Kalcher-zstd, and byte-domain zstd --train.

Both axes: compression ratio AND median decode us/chunk (decode to TOKEN IDs,
incl. LEB128 decode + rank un-permute; timed_reps median-of-30, matching
07_kalcher_baseline/bench_unified_latency.py). Dict-size sweep: 16K/64K/112K/256K.

Methodology matched to Table 1 path B (table1_full_train_consistent): seed 9012,
512-token chunks, full-train tables, cl100k (the strongest Kalcher tokenizer on
code). Dictionary trained on the ENTIRE code TRAIN split (pooled across all
train chunks); evaluated on held-out TEST chunks -- no leakage.
"""
import os
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TIKTOKEN_MAX_THREADS", "1")

import sys
import json
import time
import lzma
import numpy as np
import tiktoken
import zstandard as zstd
import constriction

# reuse the validated LEB128 / streamvbyte / LZMA helpers from 07_kalcher_baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "07_kalcher_baseline"))
from bench_kalcher import (  # noqa: E402
    leb128_encode, leb128_decode, svb_encode_arr, svb_decode_arr, LZMA_FILTERS,
)

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
CHUNK_SIZE = 512
N_CHUNKS = 40
REPS = 30
SEED = 9012                       # match Table 1 path B
N_BOOTSTRAP = 2000
TOKENIZERS = {"r50k": ("r50k_base", 50257),
              "cl100k": ("cl100k_base", 100277),
              "o200k": ("o200k_base", 200019)}
DICT_SIZES = [16 * 1024, 64 * 1024, 112 * 1024, 256 * 1024]
MAX_TRAIN_SAMPLES = 15000         # pooled train chunks used to train the dict
# code = full sweep; prose/hindi = single 112K dict as a "does it help language?" check
DOMAIN_PLAN = {"code": DICT_SIZES, "prose": [112 * 1024], "hindi": [112 * 1024]}

r50k = tiktoken.get_encoding("r50k_base")
zstd_c22 = zstd.ZstdCompressor(level=22)
zstd_d = zstd.ZstdDecompressor()


def bootstrap_ci(values, rng, n_boot=N_BOOTSTRAP, alpha=0.10):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(np.median(values)), (float(values[0]), float(values[0]))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    bm = np.median(values[idx], axis=1)
    lo, hi = np.percentile(bm, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(values)), (float(lo), float(hi))


def make_chunks(arr, chunk_size, n_chunks, rng):
    max_chunks = len(arr) // chunk_size
    n = min(n_chunks, max_chunks)
    starts = rng.choice(max_chunks, size=n, replace=False) * chunk_size
    return [arr[s: s + chunk_size] for s in starts]


def timed_reps(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e6)
    return float(np.median(ts))


def run_domain(domain, tok, enc, vocab, rng):
    print(f"\n########## {domain} ({tok}) ##########", flush=True)
    train = np.load(os.path.join(CORPUS_DIR, f"{domain}_train.npy")).astype(np.int64)
    test = np.load(os.path.join(CORPUS_DIR, f"{domain}_test.npy")).astype(np.int64)
    train_text = r50k.decode(train.tolist())

    # full-train rank table (freq remap) + ANS model, in this tokenizer
    train_ids = np.array(enc.encode(train_text, disallowed_special=()), dtype=np.int64)
    fcounts = np.bincount(train_ids, minlength=vocab)
    order = np.argsort(-fcounts)
    rank_of = np.empty(vocab, dtype=np.uint32)
    rank_of[order] = np.arange(vocab, dtype=np.uint32)
    token_of_rank = order.astype(np.int64)
    ans_counts = np.ones(vocab, dtype=np.int64) + fcounts
    ans_probs = ans_counts.astype(np.float64) / ans_counts.sum()
    ans_model = constriction.stream.model.Categorical(ans_probs, perfect=False)

    # ---- corpus-wide training samples: every 512-r50k-window of TRAIN, as
    #      freq-remapped LEB128 varint bytes (same object as a test payload) ----
    n_train_windows = len(train) // CHUNK_SIZE
    idxs = np.arange(n_train_windows)
    if n_train_windows > MAX_TRAIN_SAMPLES:
        idxs = rng.choice(n_train_windows, size=MAX_TRAIN_SAMPLES, replace=False)
    samples = []
    for i in idxs:
        w = train[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        ids = np.array(enc.encode(r50k.decode(w.tolist()), disallowed_special=()), dtype=np.int64)
        samples.append(leb128_encode(rank_of[ids]))
    print(f"  built {len(samples)} corpus-wide train samples (varint)", flush=True)

    # ---- held-out test chunks ----
    chunks = make_chunks(test, CHUNK_SIZE, N_CHUNKS, rng)
    texts = [r50k.decode(c.tolist()) for c in chunks]
    raw_lens = [len(t.encode("utf-8")) for t in texts]
    test_ids = [np.array(enc.encode(t, disallowed_special=()), dtype=np.int64) for t in texts]
    test_remapped = [rank_of[ids] for ids in test_ids]
    test_varint = [leb128_encode(rm) for rm in test_remapped]

    out = {"domain": domain, "tokenizer": tok, "methods": {}}

    def record(name, ratios, decs, extra=None):
        m = {"ratio": bootstrap_ci(ratios, rng), "decode_us": bootstrap_ci(decs, rng)}
        if extra:
            m.update(extra)
        out["methods"][name] = m
        print(f"  {name:<26} ratio={m['ratio'][0]:.2f}x  decode={m['decode_us'][0]:.1f}us"
              + (f"  dict={extra['dict_bytes']//1024}K" if extra and 'dict_bytes' in extra else ""),
              flush=True)

    # ---- reference: +freq (streamvbyte) ----
    r, d = [], []
    for ids, rm, raw in zip(test_ids, test_remapped, raw_lens):
        payload = svb_encode_arr(rm)
        assert np.array_equal(svb_decode_arr(payload, len(ids)), rm)
        r.append(raw / len(payload))
        d.append(timed_reps(lambda p=payload, n=len(ids): token_of_rank[svb_decode_arr(p, n)]))
    record("+freq (streamvbyte)", r, d)

    # ---- reference: +ANS ----
    r, d = [], []
    for ids, raw in zip(test_ids, raw_lens):
        c = constriction.stream.stack.AnsCoder()
        c.encode_reverse(ids.astype(np.int32), ans_model)
        payload = c.get_compressed().tobytes()
        r.append(raw / len(payload))
        arr = np.frombuffer(payload, dtype=np.uint32).copy()
        d.append(timed_reps(lambda a=arr, n=len(ids): constriction.stream.stack.AnsCoder(a.copy()).decode(ans_model, n)))
    record("+ANS", r, d)

    # ---- reference: Kalcher-LZMA (no dict) ----
    r, d = [], []
    for var, raw in zip(test_varint, raw_lens):
        payload = lzma.compress(var, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
        r.append(raw / len(payload))
        d.append(timed_reps(lambda p=payload: token_of_rank[leb128_decode(
            lzma.decompress(p, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS))]))
    record("Kalcher (LEB128+LZMA)", r, d)

    # ---- controlled baseline: Kalcher-zstd (no dict) ----
    r, d = [], []
    for var, raw in zip(test_varint, raw_lens):
        payload = zstd_c22.compress(var)
        r.append(raw / len(payload))
        d.append(timed_reps(lambda p=payload: token_of_rank[leb128_decode(zstd_d.decompress(p))]))
    record("Kalcher (LEB128+zstd, no dict)", r, d)

    # ---- THE EXPERIMENT: LEB128 + zstd WITH corpus-wide trained dict, swept ----
    for dsize in DOMAIN_PLAN[domain]:
        zdict_raw = zstd.train_dictionary(dsize, samples)
        zdict = zstd.ZstdCompressionDict(zdict_raw.as_bytes())
        actual = len(zdict_raw.as_bytes())
        zc = zstd.ZstdCompressor(level=22, dict_data=zdict)
        zd = zstd.ZstdDecompressor(dict_data=zdict)
        r, d = [], []
        for var, ids, rm, raw in zip(test_varint, test_ids, test_remapped, raw_lens):
            payload = zc.compress(var)
            # correctness round-trip
            dec = token_of_rank[leb128_decode(zd.decompress(payload))]
            assert np.array_equal(dec, ids)
            r.append(raw / len(payload))
            d.append(timed_reps(lambda p=payload: token_of_rank[leb128_decode(zd.decompress(p))]))
        record(f"dict-zstd ({dsize // 1024}K)", r, d, extra={"dict_bytes": actual})

    # ---- reference: byte-domain zstd --train (trained on TRAIN utf-8 chunks) ----
    byte_samples = [r50k.decode(train[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE].tolist()).encode("utf-8")
                    for i in (idxs[:4000] if len(idxs) > 4000 else idxs)]
    bdict_raw = zstd.train_dictionary(112 * 1024, byte_samples)
    bdict = zstd.ZstdCompressionDict(bdict_raw.as_bytes())
    zbc = zstd.ZstdCompressor(level=19, dict_data=bdict)
    zbd = zstd.ZstdDecompressor(dict_data=bdict)
    r, d = [], []
    for t, raw in zip(texts, raw_lens):
        b = t.encode("utf-8")
        payload = zbc.compress(b)
        r.append(raw / len(payload))
        # decode-to-token-IDs for byte domain = decompress + RE-TOKENIZE (honest)
        d.append(timed_reps(lambda p=payload: enc.encode(zbd.decompress(p).decode("utf-8"), disallowed_special=())))
    record("byte zstd --train (112K, +retokenize)", r, d,
           extra={"dict_bytes": len(bdict_raw.as_bytes()),
                  "note": "decode lands at TEXT; decode_us includes re-tokenize to reach token IDs"})

    return out


def main():
    results = {
        "config": {
            "tokenizers": list(TOKENIZERS), "chunk_size": CHUNK_SIZE, "n_chunks": N_CHUNKS,
            "reps": REPS, "seed": SEED,
            "seed_scheme": "per-cell np.random.default_rng([SEED, tok_idx, domain_idx]) so each "
                           "(tokenizer,domain) is reproducible and independent of run order",
            "methodology": "matches Table 1 path B (full-train rank/ANS); dict trained "
                           "corpus-wide on TRAIN split varint samples, evaluated on held-out TEST",
            "decode_convention": "median-of-30 reps, decode to token IDs incl. LEB128 + rank un-permute",
            "dict_sizes_code": [d // 1024 for d in DICT_SIZES],
            "max_train_samples": MAX_TRAIN_SAMPLES,
        },
        "grid": {},
    }
    for ti, (tok, (enc_name, vocab)) in enumerate(TOKENIZERS.items()):
        enc = tiktoken.get_encoding(enc_name)
        results["grid"][tok] = {}
        for di, domain in enumerate(["code", "prose", "hindi"]):
            rng = np.random.default_rng([SEED, ti, di])
            results["grid"][tok][domain] = run_domain(domain, tok, enc, vocab, rng)

    # ── summary grid: zstd --train on token IDs (112K dict) per tokenizer x domain ──
    print(f"\n{'=' * 90}\nzstd --train ON TOKEN IDs (112K dict) -- ratio per tokenizer x domain\n{'=' * 90}")
    print(f"  {'tok':<8}{'domain':<8}{'no-dict':>9}{'dict-112K':>11}{'delta':>8}{'Kalcher-LZMA':>14}")
    for tok in TOKENIZERS:
        for domain in ["code", "prose", "hindi"]:
            md = results["grid"][tok][domain]["methods"]
            nd = md["Kalcher (LEB128+zstd, no dict)"]["ratio"][0]
            dd = md["dict-zstd (112K)"]["ratio"][0]
            kl = md["Kalcher (LEB128+LZMA)"]["ratio"][0]
            print(f"  {tok:<8}{domain:<8}{nd:>8.2f}x{dd:>10.2f}x{dd - nd:>+7.2f}x{kl:>13.2f}x")

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
