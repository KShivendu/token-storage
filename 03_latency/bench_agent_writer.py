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
import time
import numpy as np
import tiktoken
import pyfastpfor
import constriction

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
DOMAINS = ["prose", "code", "hindi"]
CHUNK_SIZE = 512
N_CHUNKS = 40
REPS = 30
TOKENIZERS = {"r50k": ("r50k_base", 50257), "cl100k": ("cl100k_base", 100277), "o200k": ("o200k_base", 200019)}
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(3344)

r50k = tiktoken.get_encoding("r50k_base")
svb_codec = pyfastpfor.getCodec("streamvbyte")


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=0.10):
    values = np.asarray(values, dtype=np.float64)
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    boot_medians = np.median(values[idx], axis=1)
    lo, hi = np.percentile(boot_medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(values)), (float(lo), float(hi))


def make_chunks(test_arr, chunk_size, n_chunks):
    max_chunks = len(test_arr) // chunk_size
    n = min(n_chunks, max_chunks)
    starts = RNG.choice(max_chunks, size=n, replace=False) * chunk_size
    return [test_arr[s: s + chunk_size] for s in starts]


def timed_reps(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e6)
    return float(np.median(ts))


def pack3(ids):
    n = len(ids)
    out = np.zeros(n * 3, dtype=np.uint8)
    out[0::3] = (ids >> 16) & 0xFF
    out[1::3] = (ids >> 8) & 0xFF
    out[2::3] = ids & 0xFF
    return out.tobytes()


def svb_encode_arr(arr):
    out = np.zeros(len(arr) * 2 + 1024, dtype=np.uint32)
    n_out = svb_codec.encodeArray(arr, len(arr), out, len(out))
    return out[:n_out].tobytes()


results = {}
for domain in DOMAINS:
    print(f"=== {domain} ===", flush=True)
    train_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_train.npy")).astype(np.int64)
    test_r50k = np.load(os.path.join(CORPUS_DIR, f"{domain}_test.npy")).astype(np.int64)
    train_text = r50k.decode(train_r50k.tolist())

    chunks_r50k = make_chunks(test_r50k, CHUNK_SIZE, N_CHUNKS)
    texts = [r50k.decode(c.tolist()) for c in chunks_r50k]

    for tok_key, (enc_name, vocab_size) in TOKENIZERS.items():
        enc = tiktoken.get_encoding(enc_name)

        # frequency rank table, same discipline as bench_freqremap.py: trained
        # on this domain's train split only.
        train_ids = enc.encode(train_text, disallowed_special=())
        counts = np.zeros(vocab_size, dtype=np.int64)
        vals, cnts = np.unique(train_ids, return_counts=True)
        counts[vals] = cnts
        order = np.argsort(-counts)
        rank_of = np.empty(vocab_size, dtype=np.uint32)
        rank_of[order] = np.arange(vocab_size, dtype=np.uint32)

        ans_counts = np.ones(vocab_size, dtype=np.int64)
        ans_counts += np.bincount(train_ids, minlength=vocab_size)
        ans_probs = ans_counts.astype(np.float64) / ans_counts.sum()
        ans_model = constriction.stream.model.Categorical(ans_probs, perfect=False)

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
