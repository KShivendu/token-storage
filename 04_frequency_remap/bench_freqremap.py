"""
Frequency-sorted token IDs + streamvbyte, measured the same honest way as
bench_agent_serving.py: decode split into "agent-ready" (streamvbyte decode
+ reverse-remap lookup -> token IDs, no text) and "human tax" (detokenize
on top, only paid if/when a human reads it). Same corpus, same domains,
same chunk sizes, same RNG seed as bench_agent_serving.py so the numbers
are directly comparable, not just similar.

Encode = frequency-remap (rank lookup) + streamvbyte encode, timed together
with the tokenizer's own encode (matching how "raw"/"+ANS" encode was timed
elsewhere in this repo -- full text-to-storage-format cost).
"""
import os
import sys
import time
import numpy as np
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tnbench as T
from tnbench import svb_encode_arr, svb_decode_arr, build_rank_table, load_ids

DOMAINS = ["prose", "code", "hindi"]
CHUNK_SIZES = [256, 512, 2000]
N_CHUNKS = 40
TOKENIZERS = {"r50k": ("r50k_base", 50257), "cl100k": ("cl100k_base", 100277), "o200k": ("o200k_base", 200019)}
RNG = np.random.default_rng(3344)  # same seed as bench_agent_mode_v2.py

def bootstrap_ci(values):
    return T.bootstrap_ci(values, RNG)


def make_chunks(test_arr, chunk_size, n_chunks):
    return T.make_chunks(test_arr, chunk_size, n_chunks, RNG)


r50k = tiktoken.get_encoding("r50k_base")


results = {}
for domain in DOMAINS:
    print(f"=== {domain} ===")
    train_r50k = load_ids(f"{domain}_train")
    test_r50k = load_ids(f"{domain}_test")
    train_text = r50k.decode(train_r50k.tolist())

    for chunk_size in CHUNK_SIZES:
        chunks_r50k = make_chunks(test_r50k, chunk_size, N_CHUNKS)
        texts = [r50k.decode(c.tolist()) for c in chunks_r50k]
        raw_byte_lists = [t.encode("utf-8") for t in texts]

        for tok_key, (enc_name, vocab_size) in TOKENIZERS.items():
            enc = tiktoken.get_encoding(enc_name)
            # frequency rank from this domain's TRAIN split (same discipline
            # as every other static table in this repo)
            train_ids = enc.encode(train_text, disallowed_special=())
            rank_of, token_of_rank = build_rank_table(train_ids, vocab_size)

            ratios, enc_t, agent_dec_t, human_extra_t = [], [], [], []
            for raw, text in zip(raw_byte_lists, texts):
                t0 = time.perf_counter()
                ids = enc.encode(text, disallowed_special=())
                ids_arr = np.array(ids, dtype=np.int64)
                remapped = rank_of[ids_arr]
                payload = svb_encode_arr(remapped)
                t1 = time.perf_counter()

                t2 = time.perf_counter()
                decoded_remapped = svb_decode_arr(payload, len(ids))
                decoded_ids = token_of_rank[decoded_remapped]
                t3 = time.perf_counter()
                decoded_text = enc.decode(list(decoded_ids))
                t4 = time.perf_counter()

                assert list(decoded_ids) == ids, f"{domain}/{tok_key}/{chunk_size}: round-trip mismatch"
                assert decoded_text == text

                ratios.append(len(raw) / len(payload))
                enc_t.append(t1 - t0)
                agent_dec_t.append(t3 - t2)
                human_extra_t.append(t4 - t3)

            results[(domain, tok_key, chunk_size, "ratio")] = bootstrap_ci(ratios)
            results[(domain, tok_key, chunk_size, "encode")] = bootstrap_ci(np.array(enc_t) * 1e6)
            results[(domain, tok_key, chunk_size, "agent_decode")] = bootstrap_ci(
                np.array(agent_dec_t) * 1e6
            )
            results[(domain, tok_key, chunk_size, "human_extra_decode")] = bootstrap_ci(
                np.array(human_extra_t) * 1e6
            )
        print(f"  chunk_size={chunk_size} done")

print(f"\n{'=' * 110}")
print("  Frequency-sorted token IDs + streamvbyte -- ratio, encode, agent-decode, +human-extra (detokenize)")
print(f"{'=' * 110}")
print(f"  {'domain':<8}{'tok':>4}{'size':>6}{'ratio':>18}{'encode us':>18}{'agent-dec us':>18}{'+human us':>16}")
for domain in DOMAINS:
    for cs in CHUNK_SIZES:
        for tok_key in TOKENIZERS:
            r = results[(domain, tok_key, cs, "ratio")]
            e = results[(domain, tok_key, cs, "encode")]
            a = results[(domain, tok_key, cs, "agent_decode")]
            h = results[(domain, tok_key, cs, "human_extra_decode")]
            print(
                f"  {domain:<8}{tok_key:>4}{cs:>6}"
                f" {r[0]:.2f}x[{r[1][0]:.2f},{r[1][1]:.2f}]".rjust(19)
                + f" {e[0]:.1f}[{e[1][0]:.1f},{e[1][1]:.1f}]".rjust(19)
                + f" {a[0]:.1f}[{a[1][0]:.1f},{a[1][1]:.1f}]".rjust(19)
                + f" {h[0]:.1f}".rjust(16)
            )

import json
with open(os.path.join(os.path.dirname(__file__), "freqremap_results.json"), "w") as f:
    json.dump({f"{d}|{t}|{c}|{m}": v for (d, t, c, m), v in results.items()}, f, indent=2)
print("\nSaved to freqremap_results.json")
