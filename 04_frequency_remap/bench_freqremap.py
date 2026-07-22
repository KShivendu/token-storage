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
import time
import numpy as np
import tiktoken
import pyfastpfor

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
DOMAINS = ["prose", "code", "hindi"]
CHUNK_SIZES = [256, 512, 2000]
N_CHUNKS = 40
TOKENIZERS = {"r50k": ("r50k_base", 50257), "cl100k": ("cl100k_base", 100277), "o200k": ("o200k_base", 200019)}
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(3344)  # same seed as bench_agent_serving.py

r50k = tiktoken.get_encoding("r50k_base")
svb_codec = pyfastpfor.getCodec("streamvbyte")


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=0.10):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(np.median(values)), (float(values[0]), float(values[0]))
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    boot_medians = np.median(values[idx], axis=1)
    lo, hi = np.percentile(boot_medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(values)), (float(lo), float(hi))


def make_chunks(test_arr, chunk_size, n_chunks):
    max_chunks = len(test_arr) // chunk_size
    n = min(n_chunks, max_chunks)
    starts = RNG.choice(max_chunks, size=n, replace=False) * chunk_size
    return [test_arr[s : s + chunk_size] for s in starts]


def svb_encode_arr(arr):
    out = np.zeros(len(arr) * 2 + 1024, dtype=np.uint32)
    n_out = svb_codec.encodeArray(arr, len(arr), out, len(out))
    return out[:n_out].tobytes()


def svb_decode_arr(payload, n):
    packed = np.frombuffer(payload, dtype=np.uint32)
    out = np.zeros(n + 1024, dtype=np.uint32)
    svb_codec.decodeArray(packed, len(packed), out, n)
    return out[:n]


results = {}
for domain in DOMAINS:
    print(f"=== {domain} ===")
    train_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_train.npy")).astype(np.int64)
    test_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_test.npy")).astype(np.int64)
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
            counts = np.zeros(vocab_size, dtype=np.int64)
            vals, cnts = np.unique(train_ids, return_counts=True)
            counts[vals] = cnts
            order = np.argsort(-counts)  # token_id sorted most -> least frequent
            rank_of = np.empty(vocab_size, dtype=np.uint32)
            rank_of[order] = np.arange(vocab_size, dtype=np.uint32)
            token_of_rank = order.astype(np.int64)

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
