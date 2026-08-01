"""
Task A: Kalcher on cl100k and o200k (apples-to-apples ratio).

bench_kalcher.py ran Kalcher only on r50k. But the paper's headline
"tokenizer+ANS wins by a wide margin" uses the BEST tokenizer per language
(cl100k+ANS on code, o200k+ANS on Hindi). So this runs Kalcher
(LEB128+LZMA and LEB128+zstd-22) on r50k / cl100k / o200k for all three
domains and lines it up against each tokenizer's +ANS.

Methodology is matched to 01_storage_efficiency/bench_summary_tables.py --
the script that produced the blog's 3-domain ratio table -- so the +ANS
column reproduces the published numbers:
  - RNG seed 9012, 512-token chunks, 40 test chunks.
  - Static tables (ANS model AND the frequency rank table Kalcher/+freq share)
    trained on each domain's FULL train split, re-tokenized with the target
    tokenizer (corpus arrays are r50k IDs; detokenize r50k -> text ->
    re-tokenize, exactly as bench_summary_tables / bench_tokenizer_gen do).
  - Ratio excludes any length header (LEB128 is self-delimiting).

Reuses leb128_encode / zstd / LZMA settings from bench_kalcher.py.
"""
import os
import json
import lzma
import numpy as np
import tiktoken
import zstandard as zstd
import constriction

from bench_kalcher import (
    leb128_encode,
    leb128_decode,
    zstd_c22,
    LZMA_FILTERS,
    bootstrap_ci,
)

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
DOMAINS = ["prose", "code", "hindi"]
CHUNK_SIZE = 512
N_CHUNKS = 40
SEED = 9012  # match bench_summary_tables.py (the blog's 3-domain ratio table)
TOKENIZERS = {"r50k": ("r50k_base", 50257), "cl100k": ("cl100k_base", 100277), "o200k": ("o200k_base", 200019)}

r50k = tiktoken.get_encoding("r50k_base")


def make_chunks(test_arr, chunk_size, n_chunks, rng):
    max_chunks = len(test_arr) // chunk_size
    n = min(n_chunks, max_chunks)
    starts = rng.choice(max_chunks, size=n, replace=False) * chunk_size
    return [test_arr[s : s + chunk_size] for s in starts]


def main():
    rng = np.random.default_rng(SEED)
    results = {}  # (domain, tok, method) -> ratio ci
    train_text_cache = {}

    for domain in DOMAINS:
        print(f"=== {domain} ===", flush=True)
        train_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_train.npy")).astype(np.int64)
        test_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_test.npy")).astype(np.int64)
        if domain not in train_text_cache:
            train_text_cache[domain] = r50k.decode(train_r50k.tolist())
        train_text = train_text_cache[domain]

        chunks = make_chunks(test_r50k, CHUNK_SIZE, N_CHUNKS, rng)
        texts = [r50k.decode(c.tolist()) for c in chunks]
        raw_lens = [len(t.encode("utf-8")) for t in texts]

        for tok_key, (enc_name, vocab_size) in TOKENIZERS.items():
            enc = tiktoken.get_encoding(enc_name)

            # full-train frequency: rank table (Kalcher/+freq) + ANS model
            train_ids = np.array(enc.encode(train_text, disallowed_special=()), dtype=np.int64)
            counts = np.bincount(train_ids, minlength=vocab_size)
            order = np.argsort(-counts)
            rank_of = np.empty(vocab_size, dtype=np.uint32)
            rank_of[order] = np.arange(vocab_size, dtype=np.uint32)

            ans_counts = np.ones(vocab_size, dtype=np.int64) + counts
            ans_probs = ans_counts.astype(np.float64) / ans_counts.sum()
            ans_model = constriction.stream.model.Categorical(ans_probs, perfect=False)

            ans_r, klzma_r, kzstd_r = [], [], []
            for text, raw_len in zip(texts, raw_lens):
                ids = np.array(enc.encode(text, disallowed_special=()), dtype=np.int64)
                remapped = rank_of[ids]

                # +ANS
                c = constriction.stream.stack.AnsCoder()
                c.encode_reverse(ids.astype(np.int32), ans_model)
                ans_payload = c.get_compressed().tobytes()
                ans_r.append(raw_len / len(ans_payload))

                # Kalcher: freq-remap -> LEB128 -> {LZMA, zstd-22}
                varint = leb128_encode(remapped)
                assert np.array_equal(leb128_decode(varint), remapped)
                lzma_payload = lzma.compress(varint, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
                zstd_payload = zstd_c22.compress(varint)
                klzma_r.append(raw_len / len(lzma_payload))
                kzstd_r.append(raw_len / len(zstd_payload))

            results[(domain, tok_key, "+ANS")] = bootstrap_ci(ans_r, rng)
            results[(domain, tok_key, "Kalcher(LZMA)")] = bootstrap_ci(klzma_r, rng)
            results[(domain, tok_key, "Kalcher(zstd)")] = bootstrap_ci(kzstd_r, rng)
            print(
                f"  {tok_key:<7} +ANS={results[(domain,tok_key,'+ANS')][0]:.2f}x  "
                f"Kalcher(LZMA)={results[(domain,tok_key,'Kalcher(LZMA)')][0]:.2f}x  "
                f"Kalcher(zstd)={results[(domain,tok_key,'Kalcher(zstd)')][0]:.2f}x",
                flush=True,
            )

    # ── full per-tokenizer table ──
    print(f"\n{'='*95}")
    print("RATIO vs UTF-8 per tokenizer (median [90% CI]), 512-token chunks, seed 9012")
    print(f"{'='*95}")
    for domain in DOMAINS:
        print(f"\n-- {domain} --")
        print(f"  {'tok':<8}{'+ANS':>22}{'Kalcher(LZMA)':>22}{'Kalcher(zstd)':>22}")
        for tok_key in TOKENIZERS:
            a = results[(domain, tok_key, "+ANS")]
            l = results[(domain, tok_key, "Kalcher(LZMA)")]
            z = results[(domain, tok_key, "Kalcher(zstd)")]
            print(
                f"  {tok_key:<8}"
                + f"{a[0]:.2f}x[{a[1][0]:.2f},{a[1][1]:.2f}]".rjust(22)
                + f"{l[0]:.2f}x[{l[1][0]:.2f},{l[1][1]:.2f}]".rjust(22)
                + f"{z[0]:.2f}x[{z[1][0]:.2f},{z[1][1]:.2f}]".rjust(22)
            )

    # ── best-tokenizer-per-language comparison ──
    best_tok = {"prose": "r50k", "code": "cl100k", "hindi": "o200k"}  # best +ANS per blog
    print(f"\n{'='*95}")
    print("BEST-TOKENIZER-PER-LANGUAGE: +ANS vs Kalcher (each with its best tokenizer)")
    print(f"{'='*95}")
    print(f"  {'language':<10}{'best tok':>10}{'+ANS':>14}{'Kalcher(LZMA)':>16}{'Kalcher(zstd)':>16}")
    best_summary = {}
    for domain in DOMAINS:
        # best tokenizer for each method independently (fairest)
        ans_best_tok = max(TOKENIZERS, key=lambda t: results[(domain, t, "+ANS")][0])
        klzma_best_tok = max(TOKENIZERS, key=lambda t: results[(domain, t, "Kalcher(LZMA)")][0])
        kzstd_best_tok = max(TOKENIZERS, key=lambda t: results[(domain, t, "Kalcher(zstd)")][0])
        a = results[(domain, ans_best_tok, "+ANS")][0]
        l = results[(domain, klzma_best_tok, "Kalcher(LZMA)")][0]
        z = results[(domain, kzstd_best_tok, "Kalcher(zstd)")][0]
        print(
            f"  {domain:<10}{'':>10}"
            f"{a:.2f}x ({ans_best_tok})".rjust(14 + 9)
            + f"{l:.2f}x ({klzma_best_tok})".rjust(16 + 2)
            + f"{z:.2f}x ({kzstd_best_tok})".rjust(16 + 2)
        )
        best_summary[domain] = {
            "+ANS": {"tok": ans_best_tok, "ratio": a},
            "Kalcher(LZMA)": {"tok": klzma_best_tok, "ratio": l},
            "Kalcher(zstd)": {"tok": kzstd_best_tok, "ratio": z},
        }

    # merge into results.json
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing["taskA_multitok_ratio"] = {
        "config": {"seed": SEED, "chunk_size": CHUNK_SIZE, "n_chunks": N_CHUNKS,
                   "train": "full train split, per (domain,tokenizer), matches bench_summary_tables.py",
                   "lzma": "FORMAT_RAW LZMA2 preset=9|EXTREME", "zstd": "level 22"},
        "per_tokenizer": {f"{d}|{t}|{m}": results[(d, t, m)]
                          for d in DOMAINS for t in TOKENIZERS
                          for m in ["+ANS", "Kalcher(LZMA)", "Kalcher(zstd)"]},
        "best_tokenizer_per_language": best_summary,
    }
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print("\nMerged into results.json (taskA_multitok_ratio)")


if __name__ == "__main__":
    main()
