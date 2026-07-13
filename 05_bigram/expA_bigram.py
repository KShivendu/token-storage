"""
Experiment A — Order-1 (bigram) token model: achievable ANS compression ratio.
Runs on Modal (CPU, 32GB) — bigram counting over ~117M train tokens is too heavy
for the local machine.

Phase 1 is pure measurement, no entropy coder needed: compute the conditional
cross-entropy of WikiText-103 test articles under a pruned bigram model with
absolute-discounting backoff to the static unigram table.

    achievable ratio = 8 * utf8_bytes / total_bits

ANS gets within <1% of this, so the analytic number IS the result. Phase 2
(implementing the coder in constriction) only happens if these numbers justify it.

Model: absolute discounting (d=0.75) with backoff to static unigram:
    P(c|p) = (N(p,c) - d) / N(p)                     if (p,c) in kept table
           = alpha(p) * P_uni(c) / backnorm(p)        otherwise
    alpha(p)    = 1 - sum_kept (N(p,c)-d)/N(p)        (discount + pruned mass)
    backnorm(p) = 1 - sum_kept P_uni(c)               (renormalize backoff dist)

Pruning: keep globally top-K pairs by count, K set by table-size budget.
The unigram table is recomputed inside the container with the exact recipe from
the published benchmark (Laplace +1 on train counts), so the sanity baseline is
directly comparable to the post's 3.35x.

Run: modal run expA_bigram.py
"""

import modal

app = modal.App("expA-bigram-ans")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy", "tiktoken", "datasets"
)

TOKENIZERS = {"r50k_base": 50_257, "cl100k_base": 100_277, "o200k_base": 200_019}
D = 0.75  # absolute discount
# (label, n_kept_pairs) — assume ~6 bytes/pair stored (uint16 cur + quantized prob,
# grouped by prev); report 8 B/pair as conservative
BUDGETS = [("1MB", 175_000), ("10MB", 1_750_000), ("100MB", 17_500_000), ("full", None)]


@app.function(image=image, cpu=8, memory=32768, timeout=3600)
def run(enc_name: str = "r50k_base"):
    import time
    import numpy as np
    import tiktoken
    from datasets import load_dataset

    VOCAB = TOKENIZERS[enc_name]
    enc = tiktoken.get_encoding(enc_name)
    print(f"=== tokenizer: {enc_name} (vocab {VOCAB:,}) ===", flush=True)
    t0 = time.time()

    print("=== tokenize train split (chunked) ===", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    lines = ds["text"]
    parts, B = [], 200_000
    for i in range(0, len(lines), B):
        ids = enc.encode_ordinary("\n".join(lines[i : i + B]))
        parts.append(np.array(ids, dtype=np.int32))
    ids = np.concatenate(parts)
    del parts
    print(f"  {len(ids):,} train tokens ({time.time()-t0:.0f}s)", flush=True)

    # unigram table — exact recipe from the published benchmark (Laplace +1)
    counts = np.bincount(ids, minlength=VOCAB).astype(np.int64) + 1
    P_uni = counts.astype(np.float64) / counts.sum()
    log2_uni = np.log2(P_uni)

    ids = ids.astype(np.int64)
    keys = ids[:-1] * VOCAB + ids[1:]
    del ids
    K, C = np.unique(keys, return_counts=True)  # K sorted ascending
    del keys
    prev_all = (K // VOCAB).astype(np.int32)
    cur_all = (K % VOCAB).astype(np.int32)
    N_prev = np.bincount(prev_all, weights=C.astype(np.float64), minlength=VOCAB)
    print(f"  {len(K):,} unique pairs ({time.time()-t0:.0f}s)", flush=True)

    # ── test set: one big stream with doc boundaries ─────────────────────────
    print("=== tokenize test articles ===", flush=True)
    ds_test = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test")
    articles = [t.strip() for t in ds_test["text"] if len(t.strip().split()) >= 30]
    doc_ids = [np.array(enc.encode_ordinary(a), dtype=np.int64) for a in articles]
    doc_bytes = np.array([len(a.encode("utf-8")) for a in articles], dtype=np.float64)
    keep = [i for i, d_ in enumerate(doc_ids) if len(d_) >= 2]
    doc_ids = [doc_ids[i] for i in keep]
    doc_bytes = doc_bytes[keep]
    lens = np.array([len(d_) for d_ in doc_ids])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    all_t = np.concatenate(doc_ids)
    print(f"  {len(doc_ids)} articles, {len(all_t):,} test tokens", flush=True)

    pair_mask = np.ones(len(all_t), dtype=bool)
    pair_mask[starts] = False  # token i forms pair (i-1, i) only within a doc
    pos_idx = np.nonzero(pair_mask)[0]
    t_prev = all_t[pos_idx - 1]
    t_cur = all_t[pos_idx]
    pair_keys = t_prev * VOCAB + t_cur
    doc_of_pos = np.searchsorted(starts, pos_idx, side="right") - 1
    bits_first = -log2_uni[all_t[starts]]

    # unigram baseline (sanity check vs published 3.35x)
    uni_doc_bits = np.add.reduceat(-log2_uni[all_t], starts)
    uni_ratios = doc_bytes * 8 / uni_doc_bits
    print(f"\n[sanity] unigram analytic mean ratio: {uni_ratios.mean():.3f}x "
          f"(published ANS result: 3.35x)", flush=True)

    # ── evaluate each budget ─────────────────────────────────────────────────
    order_desc = np.argsort(-C, kind="stable")
    results = {"unigram_baseline_mean_ratio": float(uni_ratios.mean())}
    safe_N = np.maximum(N_prev, 1.0)

    print(f"\n{'budget':>8} {'pairs kept':>13} {'MB@8B':>8} {'bits/tok':>9} "
          f"{'mean ratio':>11} {'agg ratio':>10}", flush=True)
    for label, n_keep in BUDGETS:
        if n_keep is None or n_keep >= len(K):
            mask = np.ones(len(K), dtype=bool)
            n_keep = len(K)
        else:
            mask = np.zeros(len(K), dtype=bool)
            mask[order_desc[:n_keep]] = True

        kept_K = K[mask]  # boolean mask preserves sort order
        kept_C = C[mask].astype(np.float64)
        kept_prev = prev_all[mask]
        kept_cur = cur_all[mask]

        kept_mass = np.bincount(kept_prev, weights=kept_C - D, minlength=VOCAB)
        uni_kept = np.bincount(kept_prev, weights=P_uni[kept_cur], minlength=VOCAB)
        alpha = np.maximum(np.where(N_prev > 0, 1.0 - kept_mass / safe_N, 1.0), 1e-12)
        backnorm = np.maximum(1.0 - uni_kept, 1e-12)

        pos = np.searchsorted(kept_K, pair_keys)
        pos_c = np.minimum(pos, len(kept_K) - 1)
        hit = kept_K[pos_c] == pair_keys

        bits = np.empty(len(pair_keys), dtype=np.float64)
        h, m = hit, ~hit
        bits[h] = -np.log2((kept_C[pos_c[h]] - D) / safe_N[t_prev[h]])
        bits[m] = -(np.log2(alpha[t_prev[m]]) + log2_uni[t_cur[m]] - np.log2(backnorm[t_prev[m]]))

        doc_pair_bits = np.bincount(doc_of_pos, weights=bits, minlength=len(doc_ids))
        doc_total_bits = bits_first + doc_pair_bits
        ratios = doc_bytes * 8 / doc_total_bits
        agg = float(doc_bytes.sum() * 8 / doc_total_bits.sum())
        bpt = float(doc_total_bits.sum() / len(all_t))
        mb = n_keep * 8 / 1e6

        results[label] = {"pairs": int(n_keep), "MB_at_8B": mb, "bits_per_token": bpt,
                          "mean_doc_ratio": float(ratios.mean()), "aggregate_ratio": agg,
                          "pair_hit_rate": float(hit.mean())}
        print(f"{label:>8} {n_keep:>13,} {mb:>8.1f} {bpt:>9.3f} {ratios.mean():>10.3f}x "
              f"{agg:>9.3f}x   (pair hit rate {hit.mean()*100:.1f}%)", flush=True)

    print(f"\ndone in {time.time()-t0:.0f}s", flush=True)
    return results


@app.local_entrypoint()
def main(tokenizer: str = "r50k_base"):
    import json
    results = run.remote(tokenizer)
    out = __file__.replace(".py", f"_results_{tokenizer.replace('_base','')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")
    print(json.dumps(results, indent=2))
