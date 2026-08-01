"""
"Agent writer" encode split: if the writer is itself a model, the source is
already token IDs, not text. For raw/+freq/+ANS that means skipping tokenize
entirely (encode = pack/remap/entropy-code only, given ids). For byte codecs
it's the opposite: they need text, so an agent-written payload has to be
detokenized first (encode = detokenize + compress).

Measures, per (domain, tokenizer), chunk_size=512, median of 30 reps per chunk
(same robust methodology as bench_agent_mode_v2.py):
  - detokenize_only:      ids -> text (tax byte codecs pay for agent-written content)
  - raw_pack_only:        ids -> packed bytes, no tokenize (r50k uint16 / 3-byte)
  - freq_encode_only:     ids -> rank-remap + streamvbyte-encode, no tokenize
  - ans_encode_only:      ids -> ANS-encoded bytes, no tokenize
"""
import os
import sys
import numpy as np
import tiktoken
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tnbench as T
from tnbench import timed_reps, pack3, svb_encode_arr, build_rank_table, build_ans_model, load_ids

DOMAINS = ["prose", "code", "hindi"]
CHUNK_SIZE = 512
N_CHUNKS = 40
REPS = 30
TOKENIZERS = {"r50k": ("r50k_base", 50257), "cl100k": ("cl100k_base", 100277), "o200k": ("o200k_base", 200019)}
RNG = np.random.default_rng(3344)

def bootstrap_ci(values):
    return T.bootstrap_ci(values, RNG)


def make_chunks(test_arr, chunk_size, n_chunks):
    return T.make_chunks(test_arr, chunk_size, n_chunks, RNG)


r50k = tiktoken.get_encoding("r50k_base")


results = {}
for domain in DOMAINS:
    print(f"=== {domain} ===", flush=True)
    train_r50k = load_ids(f"{domain}_train")
    test_r50k = load_ids(f"{domain}_test")
    train_text = r50k.decode(train_r50k.tolist())

    chunks_r50k = make_chunks(test_r50k, CHUNK_SIZE, N_CHUNKS)
    texts = [r50k.decode(c.tolist()) for c in chunks_r50k]

    for tok_key, (enc_name, vocab_size) in TOKENIZERS.items():
        enc = tiktoken.get_encoding(enc_name)

        # frequency rank table + static ANS model, trained on this domain's
        # train split only (same discipline as every other static table).
        train_ids = enc.encode(train_text, disallowed_special=())
        rank_of, _ = build_rank_table(train_ids, vocab_size)
        ans_model = build_ans_model(train_ids, vocab_size)

        detok_t, pack_t, freq_t, ans_t = [], [], [], []
        for text in texts:
            ids = enc.encode(text, disallowed_special=())
            ids_arr = np.array(ids, dtype=np.int64)

            detok_t.append(timed_reps(lambda ids=ids: enc.decode(ids)))

            if tok_key == "r50k":
                pack_t.append(timed_reps(lambda a=ids_arr: a.astype(np.uint16).tobytes()))
            else:
                pack_t.append(timed_reps(lambda a=ids_arr: pack3(a)))

            remapped = rank_of[ids_arr]
            freq_t.append(timed_reps(lambda r=remapped: svb_encode_arr(r)))

            ids32 = np.array(ids, dtype=np.int32)

            def ans_encode_once(ids32=ids32):
                c = constriction.stream.stack.AnsCoder()
                c.encode_reverse(ids32, ans_model)
                return c.get_compressed().tobytes()

            ans_t.append(timed_reps(ans_encode_once))

        results[(domain, tok_key, "detokenize_only")] = bootstrap_ci(detok_t)
        results[(domain, tok_key, "raw_pack_only")] = bootstrap_ci(pack_t)
        results[(domain, tok_key, "freq_encode_only")] = bootstrap_ci(freq_t)
        results[(domain, tok_key, "ans_encode_only")] = bootstrap_ci(ans_t)
        print(f"  {tok_key} detok/pack/freq/ans-encode done", flush=True)

print(f"\n{'=' * 100}")
print("  Agent-writer encode components (robust, median-of-30-reps per chunk), chunk_size=512, median us")
print(f"{'=' * 100}")
for domain in DOMAINS:
    print(f"\n-- {domain} --")
    for tok_key in TOKENIZERS:
        d = results[(domain, tok_key, "detokenize_only")]
        p = results[(domain, tok_key, "raw_pack_only")]
        f = results[(domain, tok_key, "freq_encode_only")]
        a = results[(domain, tok_key, "ans_encode_only")]
        print(
            f"  {tok_key:<8} detokenize_only={d[0]:>8.2f}us  raw_pack_only={p[0]:>7.3f}us  "
            f"freq_encode_only={f[0]:>8.3f}us  ans_encode_only={a[0]:>8.3f}us"
        )

import json
with open(os.path.join(os.path.dirname(__file__), "agent_writer_results.json"), "w") as f:
    json.dump({f"{d}|{t}|{m}": v for (d, t, m), v in results.items()}, f, indent=2)
print("\nSaved to agent_writer_results.json")
