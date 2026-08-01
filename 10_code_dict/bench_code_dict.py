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
import lzma
import numpy as np
import tiktoken
import zstandard as zstd
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import (  # noqa: E402
    load_ids, make_chunks, bootstrap_ci, timed_reps, seeded_rng,
    leb128_encode, leb128_decode, svb_encode_arr, svb_decode_arr, pack3, unpack3,
    build_rank_table, build_ans_model, zstd_c22, zstd_d, LZMA_FILTERS,
)

CHUNK_SIZE = 512
N_CHUNKS = 40
REPS = 30
SEED = 9012                       # match Table 1 path B
TOKENIZERS = {"r50k": ("r50k_base", 50257),
              "cl100k": ("cl100k_base", 100277),
              "o200k": ("o200k_base", 200019)}
DICT_SIZES = [16 * 1024, 64 * 1024, 112 * 1024, 256 * 1024]
MAX_TRAIN_SAMPLES = 15000         # pooled train chunks used to train the dict
# code = full sweep; prose/hindi = single 112K dict as a "does it help language?" check
DOMAIN_PLAN = {"code": DICT_SIZES, "prose": [112 * 1024], "hindi": [112 * 1024]}

r50k = tiktoken.get_encoding("r50k_base")


def run_domain(domain, tok, enc, vocab, rng):
    print(f"\n########## {domain} ({tok}) ##########", flush=True)
    train = load_ids(f"{domain}_train")
    test = load_ids(f"{domain}_test")
    train_text = r50k.decode(train.tolist())

    # full-train rank table (freq remap) + ANS model, in this tokenizer
    train_ids = np.array(enc.encode(train_text, disallowed_special=()), dtype=np.int64)
    rank_of, token_of_rank = build_rank_table(train_ids, vocab)
    ans_model = build_ans_model(train_ids, vocab)

    fits16 = vocab <= 65536

    def pack_ids(ids):  # raw packing, same as the `raw` method (uint16 / 3-byte)
        return ids.astype(np.uint16).tobytes() if fits16 else pack3(ids)

    def unpack_ids(buf, n):
        return np.frombuffer(buf, dtype=np.uint16).astype(np.int64) if fits16 else unpack3(buf, n)

    # ---- corpus-wide training samples: every 512-r50k-window of TRAIN, built as
    #      BOTH raw packed ID bytes (the +dict dictionary) AND freq-remapped
    #      LEB128 varint bytes (the dict-freqvarint comparison dictionary) ----
    n_train_windows = len(train) // CHUNK_SIZE
    idxs = np.arange(n_train_windows)
    if n_train_windows > MAX_TRAIN_SAMPLES:
        idxs = rng.choice(n_train_windows, size=MAX_TRAIN_SAMPLES, replace=False)
    samples, samples_idbytes = [], []
    for i in idxs:
        w = train[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        ids = np.array(enc.encode(r50k.decode(w.tolist()), disallowed_special=()), dtype=np.int64)
        samples.append(leb128_encode(rank_of[ids]))
        samples_idbytes.append(pack_ids(ids))
    print(f"  built {len(samples)} corpus-wide train samples (varint + idbytes)", flush=True)

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

    # ---- COMPARISON row (old pipeline): freq-remap -> LEB128 varint -> zstd-22
    #      WITH corpus-wide trained dict. Kept only to measure how much the
    #      freq-remap+varint preprocessing helps vs plain +dict below. ----
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
        record(f"dict-freqvarint ({dsize // 1024}K)", r, d, extra={"dict_bytes": actual})

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

    # ---- +dict (the paper's method): zstd-22 with a corpus-trained dict applied
    #      DIRECTLY to the raw packed token-ID bytes (uint16 / 3-byte) -- NO
    #      freq-remap, NO varint. The token-domain parallel of byte zstd --train.
    #      Appended last so it doesn't perturb any RNG draw above. ----
    for dsize in DOMAIN_PLAN[domain]:
        zdict_raw = zstd.train_dictionary(dsize, samples_idbytes)
        zdict = zstd.ZstdCompressionDict(zdict_raw.as_bytes())
        actual = len(zdict_raw.as_bytes())
        zc = zstd.ZstdCompressor(level=22, dict_data=zdict)
        zd = zstd.ZstdDecompressor(dict_data=zdict)
        r, d = [], []
        for ids, raw in zip(test_ids, raw_lens):
            packed = pack_ids(ids)
            payload = zc.compress(packed)
            dec = unpack_ids(zd.decompress(payload), len(ids))
            assert np.array_equal(dec, ids)
            r.append(raw / len(payload))
            d.append(timed_reps(lambda p=payload, n=len(ids): unpack_ids(zd.decompress(p), n)))
        record(f"dict-idbytes ({dsize // 1024}K)", r, d, extra={"dict_bytes": actual})

    return out


def main():
    results = {
        "config": {
            "tokenizers": list(TOKENIZERS), "chunk_size": CHUNK_SIZE, "n_chunks": N_CHUNKS,
            "reps": REPS, "seed": SEED,
            "seed_scheme": "per-cell np.random.default_rng([SEED, tok_idx, domain_idx]) so each "
                           "(tokenizer,domain) is reproducible and independent of run order",
            "methodology": "matches Table 1 path B (full-train rank/ANS). +dict (dict-idbytes) = "
                           "zstd-22 with a corpus-trained dict on the RAW PACKED token-ID bytes (uint16/"
                           "3-byte), no freq-remap/varint -- the token-domain parallel of byte zstd --train. "
                           "dict-freqvarint = old comparison pipeline (freq-remap + LEB128 varint + dict). "
                           "Dicts trained corpus-wide on TRAIN windows, evaluated on held-out TEST.",
            "decode_convention": "median-of-30 reps, decode to token IDs; +dict = zstd decompress + unpack; "
                                 "dict-freqvarint = zstd decompress + LEB128 + rank un-permute",
            "dict_sizes_code": [d // 1024 for d in DICT_SIZES],
            "max_train_samples": MAX_TRAIN_SAMPLES,
        },
        "grid": {},
    }
    for ti, (tok, (enc_name, vocab)) in enumerate(TOKENIZERS.items()):
        enc = tiktoken.get_encoding(enc_name)
        results["grid"][tok] = {}
        for di, domain in enumerate(["code", "prose", "hindi"]):
            rng = seeded_rng(SEED, ti, di)
            results["grid"][tok][domain] = run_domain(domain, tok, enc, vocab, rng)

    # ── summary grid: +dict (plain, on token-ID bytes) vs old freq-remap+varint
    #    dict, per tokenizer x domain (both 112K), + Kalcher-LZMA reference ──
    print(f"\n{'=' * 96}\n+dict (zstd --train on token-ID bytes, 112K) vs old freqvarint dict -- ratio\n{'=' * 96}")
    print(f"  {'tok':<8}{'domain':<8}{'+dict':>9}{'freqvarint':>12}{'freq-remap Δ':>14}{'Kalcher-LZMA':>14}")
    for tok in TOKENIZERS:
        for domain in ["code", "prose", "hindi"]:
            md = results["grid"][tok][domain]["methods"]
            plain = md["dict-idbytes (112K)"]["ratio"][0]
            fv = md["dict-freqvarint (112K)"]["ratio"][0]
            kl = md["Kalcher (LEB128+LZMA)"]["ratio"][0]
            print(f"  {tok:<8}{domain:<8}{plain:>8.2f}x{fv:>11.2f}x{fv - plain:>+13.2f}x{kl:>13.2f}x")

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
