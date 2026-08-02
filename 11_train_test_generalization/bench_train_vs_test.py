"""
Generalization check: TRAIN-split vs held-out TEST-split compression ratio for
the token-ID compression methods (raw, +freq, +ANS, +dict).

Question: the static tables (+freq rank table, +ANS unigram model, +dict zstd
dictionary) are all fit on the TRAIN split, but Table 1 reports ratios on TEST
chunks. If a method's tables are overfit, its ratio measured back on TRAIN
chunks will be much higher than on TEST (train >> test). A method with NO
trained table (raw) is the control: any residual train-vs-test gap there is
pure split/text-distribution difference plus sampling noise, not overfitting.

This script IMPORTS every helper from tnbench and reuses the exact, committed
method implementations rather than re-deriving them:
  - raw / +freq / +ANS: full-train table building matches 07_kalcher_baseline
    path B (build_rank_table, build_ans_model over the full TRAIN split) and
    04_frequency_remap (rank-remap -> streamvbyte).
  - +dict: the paper's dict-idbytes method from 10_code_dict -- zstd-22 with a
    corpus-trained dictionary applied to the RAW PACKED token-ID bytes
    (uint16 for r50k, 3-byte for cl100k/o200k), NO freq-remap/varint. The
    dictionary is trained on the TRAIN split's 512-token windows. This is the
    highest overfit risk: a sampled TRAIN chunk is (near-)verbatim one of the
    dictionary's own training windows.

Config matched to Table 1 path B: 512-token chunks, N_CHUNKS=40, full-train
tables, seed 9012. We evaluate BOTH splits inside one run so train_ratio and
test_ratio are directly comparable. No committed results file is touched;
output goes to train_vs_test_results.json next to this script.
"""
import os

os.environ.setdefault("RAYON_NUM_THREADS", "1")

import sys
import json

import numpy as np
import tiktoken
import zstandard as zstd
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import (  # noqa: E402
    load_ids, make_chunks, seeded_rng,
    build_rank_table, build_ans_model,
    svb_encode_arr, svb_decode_arr, pack3, unpack3,
)

CHUNK_SIZE = 512
N_CHUNKS = 40
SEED = 9012                       # match Table 1 path B
DICT_SIZE = 112 * 1024            # single 112K dict per cell (paper's dict size)
MAX_TRAIN_SAMPLES = 15000         # pooled TRAIN windows used to train the dict
TOKENIZERS = {"r50k": ("r50k_base", 50257),
              "cl100k": ("cl100k_base", 100277),
              "o200k": ("o200k_base", 200019)}
DOMAINS = ["prose", "code", "hindi"]
METHODS = ["raw", "+freq", "+ANS", "+dict"]

r50k = tiktoken.get_encoding("r50k_base")


def build_tables(domain, enc, vocab):
    """Build the three TRAIN-fit static tables for one (domain, tokenizer):
    full-train frequency rank table, full-train Laplace ANS model, and the
    112K zstd dictionary over raw packed token-ID bytes of the TRAIN windows."""
    train = load_ids(f"{domain}_train")
    train_text = r50k.decode(train.tolist())
    train_ids = np.array(enc.encode(train_text, disallowed_special=()), dtype=np.int64)

    rank_of, token_of_rank = build_rank_table(train_ids, vocab)
    ans_model = build_ans_model(train_ids, vocab)

    fits16 = vocab <= 65536

    def pack_ids(ids):
        return ids.astype(np.uint16).tobytes() if fits16 else pack3(ids)

    def unpack_ids(buf, n):
        return np.frombuffer(buf, dtype=np.uint16).astype(np.int64) if fits16 else unpack3(buf, n)

    # dict training samples = raw packed ID bytes of every TRAIN 512-r50k-window
    # (chunk-aligned, exactly the windows make_chunks can later sample), matching
    # 10_code_dict's dict-idbytes construction.
    rng_dict = seeded_rng(SEED, vocab, hash(domain) % 10007, 2)
    n_windows = len(train) // CHUNK_SIZE
    idxs = np.arange(n_windows)
    if n_windows > MAX_TRAIN_SAMPLES:
        idxs = rng_dict.choice(n_windows, size=MAX_TRAIN_SAMPLES, replace=False)
    window_texts = r50k.decode_batch([train[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE].tolist() for i in idxs])
    window_ids = enc.encode_ordinary_batch(window_texts)
    samples_idbytes = [pack_ids(np.asarray(ids, dtype=np.int64)) for ids in window_ids]

    zdict = zstd.ZstdCompressionDict(zstd.train_dictionary(DICT_SIZE, samples_idbytes).as_bytes())
    zc = zstd.ZstdCompressor(level=22, dict_data=zdict)
    zd = zstd.ZstdDecompressor(dict_data=zdict)

    return {
        "rank_of": rank_of, "token_of_rank": token_of_rank, "ans_model": ans_model,
        "pack_ids": pack_ids, "unpack_ids": unpack_ids, "zc": zc, "zd": zd,
        "n_dict_windows": len(idxs),
    }


def measure_split(split_arr, tab, enc, ti, di, split_tag):
    """Median compression ratio (utf8_bytes / compressed_bytes) over N_CHUNKS
    512-token chunks sampled from `split_arr`, for all four methods. Chunk
    selection uses a fixed per-(cell, split) seed."""
    rng = seeded_rng(SEED, ti, di, 0 if split_tag == "test" else 1)
    chunks = make_chunks(split_arr, CHUNK_SIZE, N_CHUNKS, rng)
    texts = [r50k.decode(c.tolist()) for c in chunks]
    raw_lens = [len(t.encode("utf-8")) for t in texts]

    rank_of, token_of_rank = tab["rank_of"], tab["token_of_rank"]
    ans_model = tab["ans_model"]
    pack_ids, unpack_ids = tab["pack_ids"], tab["unpack_ids"]
    zc, zd = tab["zc"], tab["zd"]

    per = {m: [] for m in METHODS}
    for text, raw in zip(texts, raw_lens):
        ids = np.array(enc.encode(text, disallowed_special=()), dtype=np.int64)

        # raw: fixed-width packing, no trained table
        packed = pack_ids(ids)
        per["raw"].append(raw / len(packed))

        # +freq: full-train rank remap -> streamvbyte
        remapped = rank_of[ids]
        svb = svb_encode_arr(remapped)
        assert np.array_equal(svb_decode_arr(svb, len(ids)), remapped)
        per["+freq"].append(raw / len(svb))

        # +ANS: full-train static unigram ANS on original IDs
        c = constriction.stream.stack.AnsCoder()
        c.encode_reverse(ids.astype(np.int32), ans_model)
        per["+ANS"].append(raw / len(c.get_compressed().tobytes()))

        # +dict: zstd-22 with full-train dict over raw packed ID bytes
        payload = zc.compress(packed)
        assert np.array_equal(unpack_ids(zd.decompress(payload), len(ids)), ids)
        per["+dict"].append(raw / len(payload))

    return {m: float(np.median(v)) for m, v in per.items()}


def main():
    rows = []          # (domain, tok, method, train, test, delta, delta_pct)
    grid = {}
    for ti, (tok, (enc_name, vocab)) in enumerate(TOKENIZERS.items()):
        enc = tiktoken.get_encoding(enc_name)
        grid[tok] = {}
        for di, domain in enumerate(DOMAINS):
            print(f"[build] {domain} / {tok} ...", flush=True)
            tab = build_tables(domain, enc, vocab)
            train_arr = load_ids(f"{domain}_train")
            test_arr = load_ids(f"{domain}_test")
            test_med = measure_split(test_arr, tab, enc, ti, di, "test")
            train_med = measure_split(train_arr, tab, enc, ti, di, "train")
            grid[tok][domain] = {}
            for m in METHODS:
                tr, te = train_med[m], test_med[m]
                delta = tr - te
                pct = 100.0 * delta / te if te else 0.0
                rows.append((domain, tok, m, tr, te, delta, pct))
                grid[tok][domain][m] = {
                    "train_ratio": round(tr, 4), "test_ratio": round(te, 4),
                    "delta": round(delta, 4), "delta_pct": round(pct, 2),
                }
            print(f"  n_dict_windows={tab['n_dict_windows']}  "
                  + "  ".join(f"{m}:tr={train_med[m]:.2f}/te={test_med[m]:.2f}" for m in METHODS),
                  flush=True)

    # ── summary table ──────────────────────────────────────────────────────
    print(f"\n{'=' * 92}")
    print("  TRAIN vs TEST compression ratio (512-tok chunks, N=40, seed 9012, full-train tables)")
    print(f"{'=' * 92}")
    print(f"  {'domain':<7}{'tok':<8}{'method':<8}{'train':>9}{'test':>9}{'delta':>9}{'delta%':>9}")
    for domain in DOMAINS:
        for tok in TOKENIZERS:
            for m in METHODS:
                g = grid[tok][domain][m]
                print(f"  {domain:<7}{tok:<8}{m:<8}"
                      f"{g['train_ratio']:>8.2f}x{g['test_ratio']:>8.2f}x"
                      f"{g['delta']:>+8.2f}x{g['delta_pct']:>+8.1f}%")

    # ── verdict metrics ────────────────────────────────────────────────────
    def cells(method):
        return [r for r in rows if r[2] == method]

    max_gap = max(rows, key=lambda r: r[5])            # by absolute delta
    raw_deltas = [abs(r[5]) for r in cells("raw")]
    raw_pcts = [abs(r[6]) for r in cells("raw")]
    dict_pcts = [r[6] for r in cells("+dict")]
    ans_pcts = [r[6] for r in cells("+ANS")]
    freq_pcts = [r[6] for r in cells("+freq")]

    print(f"\n{'=' * 92}")
    print("  VERDICT METRICS")
    print(f"{'=' * 92}")
    print(f"  raw   max |delta|={max(raw_deltas):.3f}x  max |delta%|={max(raw_pcts):.2f}%  "
          f"(control: no trained table)")
    print(f"  +freq delta% range [{min(freq_pcts):+.1f}%, {max(freq_pcts):+.1f}%]")
    print(f"  +ANS  delta% range [{min(ans_pcts):+.1f}%, {max(ans_pcts):+.1f}%]")
    print(f"  +dict delta% range [{min(dict_pcts):+.1f}%, {max(dict_pcts):+.1f}%]")
    print(f"  LARGEST gap overall: {max_gap[0]}/{max_gap[1]}/{max_gap[2]}  "
          f"train={max_gap[3]:.2f}x test={max_gap[4]:.2f}x  "
          f"delta={max_gap[5]:+.2f}x ({max_gap[6]:+.1f}%)")

    out = {
        "config": {
            "chunk_size": CHUNK_SIZE, "n_chunks": N_CHUNKS, "seed": SEED,
            "dict_size_bytes": DICT_SIZE, "max_train_samples": MAX_TRAIN_SAMPLES,
            "methods": METHODS, "tokenizers": list(TOKENIZERS), "domains": DOMAINS,
            "statistic": "median ratio = utf8_bytes / compressed_bytes",
            "note": "reuses tnbench helpers; +dict = 10_code_dict dict-idbytes; "
                    "+freq = 04_frequency_remap; +ANS/raw = 07 path B. Independent "
                    "RNG seeds, so absolute test numbers may differ marginally from "
                    "committed files -- train and test here are apples-to-apples.",
        },
        "grid": grid,
        "verdict_metrics": {
            "raw_max_abs_delta": round(max(raw_deltas), 4),
            "raw_max_abs_delta_pct": round(max(raw_pcts), 2),
            "dict_delta_pct_range": [round(min(dict_pcts), 2), round(max(dict_pcts), 2)],
            "ans_delta_pct_range": [round(min(ans_pcts), 2), round(max(ans_pcts), 2)],
            "freq_delta_pct_range": [round(min(freq_pcts), 2), round(max(freq_pcts), 2)],
            "largest_gap": {
                "domain": max_gap[0], "tokenizer": max_gap[1], "method": max_gap[2],
                "train_ratio": round(max_gap[3], 4), "test_ratio": round(max_gap[4], 4),
                "delta": round(max_gap[5], 4), "delta_pct": round(max_gap[6], 2),
            },
        },
    }
    out_path = os.path.join(os.path.dirname(__file__), "train_vs_test_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
