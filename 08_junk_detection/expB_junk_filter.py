"""
Experiment B — Per-document entropy under the static token table as a junk filter.

The ANS encoder already computes -log2 P(token) for every token it codes; the
per-document mean (bits/token) is a free byproduct of compression. Question:
does it separate clean text from junk well enough to be an ingest-time filter?

Clean: WikiText-103 test articles (the same 1,773 used in the published bench).
Probe classes:
  junk      — base64, hex dumps, keyboard mash, synthetic CJK (OOD for r50k)
  spam-ish  — lorem ipsum, repeated marketing spam (common tokens! expect MISS)
  gray      — HTML boilerplate, Python code (should NOT be flagged hard;
              false-positive check)

Baseline: per-doc zstd-3 compression ratio (what you could compute without tokens).
Metric: AUC (Mann-Whitney) of each probe class vs clean docs.
Light enough to run locally: ~3k small docs, numpy lookups only.
"""

import base64 as b64
import json
import os
import random
import zstandard as zstd
import numpy as np
import tiktoken

CACHE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(42)
nprng = np.random.default_rng(42)

enc = tiktoken.get_encoding("r50k_base")
P_uni = np.load(os.path.join(CACHE, "static_probs_r50k.npy"))
log2_uni = np.log2(P_uni)
zc = zstd.ZstdCompressor(level=3)

N_PER_CLASS = 150


def bits_per_token(text):
    ids = enc.encode_ordinary(text)
    if not ids:
        return None
    return float(-log2_uni[np.array(ids)].mean())


def zstd_ratio(text):
    raw = text.encode("utf-8")
    return len(raw) / len(zc.compress(raw))


# ── clean docs ────────────────────────────────────────────────────────────────

from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-103-v1", split="test", trust_remote_code=True)
clean = [t.strip() for t in ds["text"] if len(t.strip().split()) >= 30]
clean_lens = [len(c.encode()) for c in clean]


def target_len():
    return rng.choice(clean_lens)


# ── junk generators (length-matched to clean dist) ────────────────────────────

def gen_base64():
    n = target_len()
    return b64.b64encode(nprng.bytes(int(n * 0.75))).decode()[:n]

def gen_hex():
    n = target_len()
    raw = nprng.bytes(n // 2).hex()
    return "\n".join(raw[i:i+64] for i in range(0, len(raw), 64))

QWERTY = "qwertyuiopasdfghjklzxcvbnm"
def gen_mash():
    n = target_len()
    out, count = [], 0
    while count < n:
        w = "".join(rng.choice(QWERTY) for _ in range(rng.randint(2, 12)))
        out.append(w); count += len(w) + 1
    return " ".join(out)

LOREM = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
         "incididunt ut labore et dolore magna aliqua enim ad minim veniam quis nostrud "
         "exercitation ullamco laboris nisi aliquip ex ea commodo consequat").split()
def gen_lorem():
    n = target_len()
    words = []
    count = 0
    while count < n:
        w = rng.choice(LOREM)
        words.append(w); count += len(w) + 1
    return " ".join(words)

def gen_spam():
    n = target_len()
    phrases = ["Buy now!", "Limited offer!!", "Click here to win",
               "Best price guaranteed.", "Don't miss out -", "Act today!"]
    out, count = [], 0
    while count < n:
        p = rng.choice(phrases)
        out.append(p); count += len(p) + 1
    return " ".join(out)

def gen_cjk():
    # synthetic: uniform random CJK ideographs — labelled as such (not real Chinese)
    n = target_len()
    n_chars = n // 3  # ~3 utf-8 bytes per ideograph
    return "".join(chr(rng.randint(0x4E00, 0x9FFF)) for _ in range(n_chars))

def gen_html():
    n = target_len()
    out, count = [], 0
    while count < n:
        cid = rng.randint(1000, 9999)
        block = (f'<div class="col-md-{rng.randint(1,12)} item-{cid}" id="el{cid}">'
                 f'<span style="margin:0 auto;padding:{rng.randint(0,32)}px">'
                 f'item {cid}</span></div>')
        out.append(block); count += len(block) + 1
    return "\n".join(out)

def load_code_docs():
    """Real Python from this repo — gray class: ideally NOT flagged."""
    docs = []
    for root, _, files in os.walk(os.path.dirname(CACHE)):
        if "__pycache__" in root:
            continue
        for fn in files:
            if fn.endswith(".py"):
                try:
                    src = open(os.path.join(root, fn)).read()
                except OSError:
                    continue
                # split into chunks roughly matching clean doc sizes
                for i in range(0, len(src), 1500):
                    chunk = src[i:i+1500]
                    if len(chunk) > 400:
                        docs.append(chunk)
    rng.shuffle(docs)
    return docs[:N_PER_CLASS]


CLASSES = {
    "base64": [gen_base64() for _ in range(N_PER_CLASS)],
    "hex_dump": [gen_hex() for _ in range(N_PER_CLASS)],
    "keyboard_mash": [gen_mash() for _ in range(N_PER_CLASS)],
    "synthetic_cjk": [gen_cjk() for _ in range(N_PER_CLASS)],
    "lorem_ipsum": [gen_lorem() for _ in range(N_PER_CLASS)],
    "repeat_spam": [gen_spam() for _ in range(N_PER_CLASS)],
    "html_boilerplate": [gen_html() for _ in range(N_PER_CLASS)],
    "python_code": load_code_docs(),
}

# ── score ─────────────────────────────────────────────────────────────────────

def auc(pos, neg):
    """Mann-Whitney AUC: P(pos > neg) with tie correction."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j+1]] = ranks[order[i:j+1]].mean()
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


clean_bpt = np.array([b for b in (bits_per_token(c) for c in clean) if b is not None])
clean_zr = np.array([zstd_ratio(c) for c in clean])

print(f"clean (WikiText test, n={len(clean_bpt)}): "
      f"bits/token {clean_bpt.mean():.2f} ± {clean_bpt.std():.2f} "
      f"(p5={np.percentile(clean_bpt,5):.2f}, p95={np.percentile(clean_bpt,95):.2f})")
print(f"\n{'class':<18} {'n':>4} {'bits/tok':>14} {'AUC (entropy)':>14} {'AUC (zstd)':>11}")
print("-" * 68)

results = {"clean": {"mean_bpt": float(clean_bpt.mean()), "std": float(clean_bpt.std()),
                     "p95": float(np.percentile(clean_bpt, 95))}}
for name, docs in CLASSES.items():
    bpt = np.array([b for b in (bits_per_token(d) for d in docs) if b is not None])
    zr = np.array([zstd_ratio(d) for d in docs])
    # entropy filter: junk = HIGH bits/token → AUC of bpt
    a_ent = auc(bpt, clean_bpt)
    # zstd filter: junk = LOW compression ratio → AUC of -ratio
    a_z = auc(-zr, -clean_zr)
    results[name] = {"n": len(bpt), "mean_bpt": float(bpt.mean()),
                     "std_bpt": float(bpt.std()), "auc_entropy": float(a_ent),
                     "auc_zstd": float(a_z)}
    print(f"{name:<18} {len(bpt):>4} {bpt.mean():>8.2f} ± {bpt.std():<4.2f} "
          f"{a_ent:>13.3f} {a_z:>11.3f}")

with open(os.path.join(CACHE, "expB_junk_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved expB_junk_results.json")
print("\nNote: AUC ~1.0 = perfectly separable from clean; ~0.5 = indistinguishable;")
print("<0.5 = scores LOWER than clean (entropy filter cannot see it).")
