"""
Three supporting analyses for the paper, re-run on the multi-domain corpora
(prose=C4, code=codeparrot, hindi=mc4-hi) instead of WikiText-103, so all paper
numbers are on the same corpora:

  1. decorrelation  -- byte order-0/order-1 vs token order-0 (bits/byte, held-out).
                       Bytes are recovered exactly by detokenizing the r50k IDs.
  2. ngram          -- interpolated unigram/bigram/trigram token models (held-out).
  3. ood            -- cross-domain out-of-distribution gate: a static unigram
                       table trained on one domain, scored (bits/token) on all
                       three test sets; AUC of in-domain vs out-of-domain chunks.
                       (Real OOD, not synthetic junk.)

Reads ../data/corpus/{prose,code,hindi}_{train,test}.npy (r50k token IDs, uint32).
Run:  python 06_decorr_ngram_ood/bench.py
"""
import os, sys, json
import numpy as np
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tnbench import load_ids as load

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAINS = ["prose", "code", "hindi"]
V = 50257
enc = tiktoken.get_encoding("r50k_base")

def to_bytes(ids, cap=None):
    if cap: ids = ids[:cap]
    return enc.decode(ids.tolist()).encode("utf-8")

# ── 1. decorrelation ──────────────────────────────────────────────────────────
def decorrelation():
    print("\n=== decorrelation (held-out bits/byte; lower is better) ===")
    print(f"{'domain':<8}{'byte o0':>9}{'byte o1':>9}{'token o0':>10}   (token o0 ratio)")
    out = {}
    for d in DOMAINS:
        tr, te = load(f"{d}_train"), load(f"{d}_test")
        # byte model (held-out): train on detok(train) bytes, eval on detok(test) bytes
        a = np.frombuffer(to_bytes(tr, cap=4_000_000), np.uint8)   # ~a few MB of bytes
        t = np.frombuffer(to_bytes(te), np.uint8)
        uni = np.bincount(a, minlength=256).astype(np.float64) + 1; puni = uni / uni.sum()
        pair = a[:-1].astype(np.int64) * 256 + a[1:]
        pcond = (np.bincount(pair, minlength=65536).astype(np.float64) + 1).reshape(256, 256)
        pcond /= pcond.sum(1, keepdims=True)
        H0b = -np.mean(np.log2(puni[t]))
        H1b = -np.mean(np.log2(pcond[t[:-1], t[1:]]))
        # token order-0: unigram trained on train IDs, cross-entropy on test IDs, -> bits/byte
        tuni = np.bincount(tr, minlength=V).astype(np.float64) + 1; tp = tuni / tuni.sum()
        H0tok_bpt = -np.mean(np.log2(tp[te]))
        tok_per_byte = len(te) / len(t)
        H0tok = H0tok_bpt * tok_per_byte
        out[d] = dict(byte_o0=H0b, byte_o1=H1b, token_o0=H0tok)
        print(f"{d:<8}{H0b:>9.2f}{H1b:>9.2f}{H0tok:>10.2f}   ({8/H0tok:.2f}x)")
    return out

# ── 2. ngram ──────────────────────────────────────────────────────────────────
def ngram():
    print("\n=== n-gram token models (held-out, interpolated; ratio vs UTF-8) ===")
    print(f"{'domain':<8}{'unigram':>9}{'bigram':>9}{'trigram':>9}   (bigram/trigram context hit%)")
    out = {}
    for d in DOMAINS:
        tr, te = load(f"{d}_train"), load(f"{d}_test")
        tb = len(to_bytes(te))
        uni = np.bincount(tr, minlength=V).astype(np.float64); N = uni.sum()
        p_uni = (uni + 1) / (N + V)
        k2 = tr[:-1] * V + tr[1:]; u2, c2 = np.unique(k2, return_counts=True); del k2
        k3 = tr[:-2] * (V * V) + tr[1:-1] * V + tr[2:]; u3, c3 = np.unique(k3, return_counts=True); del k3
        def lk(u, c, keys):
            i = np.searchsorted(u, keys); i = np.clip(i, 0, len(u) - 1)
            return np.where(u[i] == keys, c[i], 0).astype(np.float64)
        a, b, c = te[:-2], te[1:-1], te[2:]
        cnt_ab = lk(u2, c2, a * V + b); cnt_b = uni[b]
        cnt_abc = lk(u3, c3, a * (V * V) + b * V + c); cnt_bc = lk(u2, c2, b * V + c)
        puc = p_uni[c]
        pb = np.where(cnt_b > 0, cnt_bc / np.maximum(cnt_b, 1), 0.0)
        pt = np.where(cnt_ab > 0, cnt_abc / np.maximum(cnt_ab, 1), 0.0)
        def bits(w1, w2, w3):
            W1 = np.full(len(c), w1); W2 = np.where(cnt_b > 0, w2, 0.0); W3 = np.where(cnt_ab > 0, w3, 0.0)
            Z = W1 + W2 + W3
            return float(-np.log2((W1 * puc + W2 * pb + W3 * pt) / Z).sum())
        r_uni = tb * 8 / bits(1, 0, 0); r_bi = tb * 8 / bits(.15, .85, 0); r_tri = tb * 8 / bits(.1, .3, .6)
        out[d] = dict(unigram=r_uni, bigram=r_bi, trigram=r_tri)
        print(f"{d:<8}{r_uni:>8.2f}x{r_bi:>8.2f}x{r_tri:>8.2f}x   ({100*(cnt_b>0).mean():.0f}% / {100*(cnt_ab>0).mean():.0f}%)")
    return out

# ── 3. ood (cross-domain) ─────────────────────────────────────────────────────
def ood():
    print("\n=== OOD gate: bits/token under each domain's static table (cross-domain) ===")
    # per-domain static unigram table (train), and per-chunk bits/token on each test set
    tables = {}
    for d in DOMAINS:
        tr = load(f"{d}_train"); uni = np.bincount(tr, minlength=V).astype(np.float64) + 1
        tables[d] = np.log2(uni / uni.sum())      # logp per token id
    CHUNK = 512
    chunks = {}   # domain -> array of chunk mean bits/token under its OWN... we need per (table, evaldomain)
    tests = {d: load(f"{d}_test") for d in DOMAINS}
    def chunk_bpt(ids, logp):
        n = (len(ids) // CHUNK) * CHUNK
        m = (-logp[ids[:n]]).reshape(-1, CHUNK).mean(1)   # mean bits/token per chunk
        return m
    print(f"{'table':<8}" + "".join(f"{('eval:'+d):>12}" for d in DOMAINS))
    mean_bpt = {}
    for td in DOMAINS:
        row = {}
        for ed in DOMAINS:
            m = chunk_bpt(tests[ed], tables[td]); row[ed] = m
        mean_bpt[td] = {ed: float(np.mean(v)) for ed, v in row.items()}
        print(f"{td:<8}" + "".join(f"{np.mean(row[ed]):>11.1f} " for ed in DOMAINS))
        chunks[td] = row
    # AUC: for each table domain, in-domain (its own test) vs each out-of-domain test
    print("\n  AUC (in-domain vs out-of-domain chunk bits/token; 1.0 = perfectly separable):")
    aucs = {}
    for td in DOMAINS:
        ins = chunks[td][td]
        for ed in DOMAINS:
            if ed == td: continue
            outs = chunks[td][ed]
            # AUC = P(out > in) since OOD scores higher bits/token
            auc = np.mean(outs[:, None] > ins[None, :]) + 0.5 * np.mean(outs[:, None] == ins[None, :])
            aucs[f"{td}_vs_{ed}"] = float(auc)
            print(f"    table={td:<6} clean={td:<6} ood={ed:<6}  AUC={auc:.3f}")
    return dict(mean_bpt=mean_bpt, auc=aucs)

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if cfg in ("decorr", "all"): res["decorr"] = decorrelation()
    if cfg in ("ngram", "all"): res["ngram"] = ngram()
    if cfg in ("ood", "all"): res["ood"] = ood()
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nsaved -> {os.path.join(HERE, 'results.json')}")
