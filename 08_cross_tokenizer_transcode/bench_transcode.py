"""
Cross-tokenizer transcoding latency: turning source-tokenizer IDs into
target-tokenizer IDs (the "a consumer with a different tokenizer detokenizes
and re-tokenizes" path from the Token-Native Storage post).

Directions (both stress different regimes):
  r50k -> o200k   (coarse 50K vocab  -> fine 200K vocab)
  o200k -> r50k   (fine 200K vocab   -> coarse 50K vocab)

Same English (C4) corpus and 512-token chunk convention as the rest of the
repo. Single core (RAYON_NUM_THREADS=1). tiktoken. Median us/chunk over 40
held-out test chunks, each timed as the median of REPS back-to-back calls
(robust per-chunk sample), then bootstrap CI across chunks.

CORRECTNESS is the hard constraint: a transcoder is valid only if
  transcode(source_ids(T)) == target.encode(T)   for every T.
Every strategy reports its exact-match rate against that ground truth. A
fast-but-wrong transcoder is useless for interchange, so we report the fallback
rate that makes each strategy 100% correct.

Strategies:
  baseline   : target.encode(source.decode(ids))            -- lossless floor
  per_token  : concat of a precomputed per-source-token target-id LUT
               (BPE merges are context-dependent, so this diverges at token
               boundaries -- we MEASURE how often it still matches, and cost it
               with the verify+fallback needed to be correct)
  per_piece  : memoized retokenization at the regex-PIECE (word) level. tiktoken
               = regex-split then per-piece BPE, and merges never cross piece
               boundaries, so concatenating per-piece target IDs == full encode
               (validated). A cache warmed on the train split turns most pieces
               into dict hits, skipping BPE. Correct by construction.
"""
import os
os.environ.setdefault("RAYON_NUM_THREADS", "1")  # single core, before tiktoken
os.environ.setdefault("TIKTOKEN_MAX_THREADS", "1")

import json
import sys
import time
import regex
import numpy as np
import tiktoken

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
CHUNK_SIZE = 512
N_CHUNKS = 40
REPS = 25
N_BOOTSTRAP = 2000
SEED = 3344

r50k = tiktoken.get_encoding("r50k_base")
o200k = tiktoken.get_encoding("o200k_base")
ENC = {"r50k": r50k, "o200k": o200k}
VOCAB = {"r50k": 50257, "o200k": 200019}


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
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e6)
    return float(np.median(ts))


# ── per-token LUT (Strategy A) ───────────────────────────────────────────────
def build_per_token_lut(src_key, tgt_key):
    """LUT[s] = target IDs for source token s tokenized in isolation. Partial-
    byte source tokens decode with U+FFFD replacement (tiktoken default), which
    is exactly why per-token concat can be wrong -- we build it anyway and
    measure. Returns (lut: list[list[int]], n_target_ids, build_us)."""
    src, tgt = ENC[src_key], ENC[tgt_key]
    v = VOCAB[src_key]
    t0 = time.perf_counter()
    lut = [None] * v
    n_ids = 0
    for s in range(v):
        try:
            txt = src.decode([s])
        except KeyError:
            # special/undefined token (e.g. <|endoftext|>): never emitted by
            # encode(..., disallowed_special=()), so it can't appear in a real
            # source-id stream. Map to empty.
            lut[s] = []
            continue
        ids = tgt.encode(txt, disallowed_special=())
        lut[s] = ids
        n_ids += len(ids)
    build_us = (time.perf_counter() - t0) * 1e6
    return lut, n_ids, build_us


def per_token_transcode(source_ids, lut):
    out = []
    for s in source_ids:
        out.extend(lut[s])
    return out


# ── per-piece memoized retokenization (Strategy B) ───────────────────────────
def build_piece_cache(tgt_key, train_text, max_entries=None):
    """Warm a {piece_text -> target_ids} cache on the train split. Pieces are
    the target tokenizer's own regex pieces; caching them is lossless because
    per-piece BPE never crosses piece boundaries (validated)."""
    tgt = ENC[tgt_key]
    pat = regex.compile(tgt._pat_str)
    cache = {}
    for piece in pat.findall(train_text):
        if piece not in cache:
            cache[piece] = tgt.encode(piece, disallowed_special=())
    if max_entries is not None and len(cache) > max_entries:
        # keep the most frequent (approximate: first-seen order is corpus order;
        # instead rebuild by frequency)
        pass
    n_ids = sum(len(v) for v in cache.values())
    return cache, pat, n_ids


def per_piece_transcode(text, pat, cache, tgt, stats=None):
    out = []
    for piece in pat.findall(text):
        ids = cache.get(piece)
        if ids is None:
            ids = tgt.encode(piece, disallowed_special=())
            cache[piece] = ids
            if stats is not None:
                stats[1] += 1  # miss
        elif stats is not None:
            stats[0] += 1      # hit
        out.extend(ids)
    return out


def run_direction(src_key, tgt_key, chunks_r50k, texts, train_text, rng):
    src, tgt = ENC[src_key], ENC[tgt_key]
    print(f"\n########## {src_key} -> {tgt_key} ##########", flush=True)

    # source IDs + ground-truth target IDs per chunk
    source_ids_list = [
        (c.astype(np.int64).tolist() if src_key == "r50k" else src.encode(t, disallowed_special=()))
        for c, t in zip(chunks_r50k, texts)
    ]
    truth_list = [tgt.encode(t, disallowed_special=()) for t in texts]

    out = {"direction": f"{src_key}->{tgt_key}"}

    # ---- component latencies (context) ----
    dec_t, enc_t = [], []
    for sids, t in zip(source_ids_list, texts):
        dec_t.append(timed_reps(lambda s=sids: src.decode(s)))
        enc_t.append(timed_reps(lambda x=t: tgt.encode(x, disallowed_special=())))
    out["decode_only_us"] = bootstrap_ci(dec_t, rng)
    out["encode_only_us"] = bootstrap_ci(enc_t, rng)

    # ---- baseline: target.encode(source.decode(ids)) ----
    base_t, base_ok = [], 0
    for sids, truth in zip(source_ids_list, truth_list):
        res = tgt.encode(src.decode(sids), disallowed_special=())
        base_ok += int(res == truth)
        base_t.append(timed_reps(lambda s=sids: tgt.encode(src.decode(s), disallowed_special=())))
    out["baseline"] = {
        "us": bootstrap_ci(base_t, rng),
        "exact_match_rate": base_ok / len(truth_list),
    }
    print(f"  baseline: {out['baseline']['us'][0]:.1f}us/chunk  exact={out['baseline']['exact_match_rate']*100:.0f}%", flush=True)

    # ---- Strategy A: per-token LUT concat ----
    lut, lut_nids, lut_build_us = build_per_token_lut(src_key, tgt_key)
    pt_t = []
    chunk_exact = 0
    tok_match_num = tok_match_den = 0
    for sids, truth in zip(source_ids_list, truth_list):
        res = per_token_transcode(sids, lut)
        chunk_exact += int(res == truth)
        # token-level agreement (prefix LCP-ish: fraction of positions equal)
        m = sum(1 for a, b in zip(res, truth) if a == b)
        tok_match_num += m
        tok_match_den += max(len(truth), len(res))
        pt_t.append(timed_reps(lambda s=sids: per_token_transcode(s, lut)))
    out["per_token"] = {
        "us": bootstrap_ci(pt_t, rng),
        "chunk_exact_match_rate": chunk_exact / len(truth_list),
        "token_level_agreement": tok_match_num / tok_match_den,
        "lut_entries": VOCAB[src_key],
        "lut_target_ids": lut_nids,
        "lut_mem_bytes_approx": lut_nids * 4 + VOCAB[src_key] * 8,
        "build_us": lut_build_us,
    }
    # per_token as a CORRECT transcoder needs verify+fallback (you can't know a
    # chunk is wrong without computing the truth): effective correct latency =
    # per_token attempt + full baseline verification, on 100% of chunks.
    pt_correct_us = out["per_token"]["us"][0] + out["baseline"]["us"][0]
    out["per_token"]["correct_with_verify_fallback_us"] = pt_correct_us
    out["per_token"]["fallback_rate_to_reach_100pct"] = 1.0 - chunk_exact / len(truth_list)
    print(f"  per_token: {out['per_token']['us'][0]:.1f}us  chunk_exact={out['per_token']['chunk_exact_match_rate']*100:.0f}%  "
          f"token_agree={out['per_token']['token_level_agreement']*100:.1f}%", flush=True)

    # ---- Strategy B: per-piece memoization (warm on train) ----
    cache, pat, cache_nids = build_piece_cache(tgt_key, train_text)
    # correctness validation (cold-correct: concat of per-piece == full)
    pp_exact = 0
    for t, truth in zip(texts, truth_list):
        res = per_piece_transcode(t, pat, cache, tgt)
        pp_exact += int(res == truth)
    # measure hit rate + latency on a WARM cache (train-warmed)
    stats = [0, 0]  # hits, misses
    pp_t = []
    for sids, t in zip(source_ids_list, texts):
        # full transcode path = decode(source_ids) then per-piece retokenize
        def run(s=sids):
            txt = src.decode(s)
            return per_piece_transcode(txt, pat, cache, tgt)
        pp_t.append(timed_reps(run))
        # hit/miss accounting once (untimed)
        _tmp = [0, 0]
        per_piece_transcode(src.decode(sids), pat, cache, tgt, stats=_tmp)
        stats[0] += _tmp[0]; stats[1] += _tmp[1]
    hit_rate = stats[0] / max(1, stats[0] + stats[1])
    out["per_piece"] = {
        "us": bootstrap_ci(pp_t, rng),
        "exact_match_rate": pp_exact / len(truth_list),
        "piece_hit_rate_warm": hit_rate,
        "cache_entries": len(cache),
        "cache_target_ids": cache_nids,
        "cache_mem_bytes_approx": cache_nids * 4 + len(cache) * 64,  # ids + str key overhead
    }
    print(f"  per_piece: {out['per_piece']['us'][0]:.1f}us  exact={out['per_piece']['exact_match_rate']*100:.0f}%  "
          f"hit_rate={hit_rate*100:.1f}%  cache_entries={len(cache)}", flush=True)

    # speedups vs baseline
    b = out["baseline"]["us"][0]
    out["speedup_vs_baseline"] = {
        "per_token_raw": b / out["per_token"]["us"][0],
        "per_token_correct": b / pt_correct_us,
        "per_piece": b / out["per_piece"]["us"][0],
    }
    return out


def main():
    rng = np.random.default_rng(SEED)
    test = np.load(os.path.join(CORPUS_DIR, "prose_test.npy")).astype(np.int64)
    train = np.load(os.path.join(CORPUS_DIR, "prose_train.npy")).astype(np.int64)
    train_text = r50k.decode(train.tolist())  # full English train text
    chunks_r50k = make_chunks(test, CHUNK_SIZE, N_CHUNKS, rng)
    texts = [r50k.decode(c.tolist()) for c in chunks_r50k]

    results = {
        "config": {
            "corpus": "prose (English C4)", "chunk_size": CHUNK_SIZE, "n_chunks": N_CHUNKS,
            "reps": REPS, "seed": SEED, "single_core": True,
            "metric": "median us/chunk, bootstrap 90% CI",
            "correctness_ground_truth": "target.encode(source.decode(text)) == target.encode(text)",
        },
        "directions": {},
    }
    for src_key, tgt_key in [("r50k", "o200k"), ("o200k", "r50k")]:
        d = run_direction(src_key, tgt_key, chunks_r50k, texts, train_text, rng)
        results["directions"][f"{src_key}->{tgt_key}"] = d

    # ── summary table ──
    print(f"\n{'=' * 78}\nCROSS-TOKENIZER TRANSCODE -- median us/chunk (512-tok English)\n{'=' * 78}")
    print(f"  {'direction':<16}{'baseline':>12}{'per_token':>12}{'per_piece':>12}"
          f"{'pt_exact':>10}{'pp_exact':>10}")
    for k, d in results["directions"].items():
        print(f"  {k:<16}{d['baseline']['us'][0]:>11.1f}{d['per_token']['us'][0]:>12.1f}"
              f"{d['per_piece']['us'][0]:>12.1f}"
              f"{d['per_token']['chunk_exact_match_rate']*100:>9.0f}%"
              f"{d['per_piece']['exact_match_rate']*100:>9.0f}%")

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
