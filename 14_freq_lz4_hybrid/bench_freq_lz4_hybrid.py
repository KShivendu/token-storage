"""Can frequency remapping and LZ4 be combined, and does it beat either alone?

`+freq` is order-0: it reassigns token IDs by frequency rank so common tokens
get small numbers, then streamvbyte-packs them. It models no repetition at all,
which is why it falls behind on code, where the same import lines and function
signatures recur constantly.

`+lz4` is the opposite: LZ4 over raw packed IDs finds repeats but ignores the
frequency skew entirely.

Stacking them is not obviously a win, because they fight:

  - Remapping SHRINKS the stream, which is good on its own, but LZ4 has a
    4-byte minimum match (MINMATCH), so a repeated token n-gram that used to
    span 12 bytes might now span 3 and stop qualifying as a match at all.
  - Variable-width encodings (streamvbyte, LEB128) also destroy byte alignment:
    the same token pair can land at different phases, so byte-level matches
    that exist logically are invisible to LZ4.
  - Fixed-width remapped IDs keep alignment AND get a bonus -- with uint16
    little-endian, any rank under 256 has a zero high byte, so the stream
    becomes lo,0,lo,0,... which LZ4 eats.

So: four ways to combine them, against the three existing baselines.

Run:  uv run python 14_freq_lz4_hybrid/bench_freq_lz4_hybrid.py
"""

import json
import sys
from pathlib import Path

import lz4.frame as lz4f
import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tnbench import (  # noqa: E402
    bootstrap_ci,
    build_ans_model,
    build_rank_table,
    leb128_decode,
    leb128_encode,
    load_ids,
    make_chunks,
    pack3,
    svb_decode_arr,
    svb_encode_arr,
    timed_reps,
    unpack3,
)

CHUNK = 512
N_CHUNKS = 60
SEED = 7788
NATIVE = {"prose": ("r50k", "r50k_base"), "code": ("cl100k", "cl100k_base"),
          "hindi": ("o200k", "o200k_base")}
OUT = Path(__file__).resolve().parent / "results.json"


def fixed_pack(arr, fits16):
    """Byte-aligned packing: uint16 when the vocabulary allows, else 3-byte."""
    return arr.astype("<u2").tobytes() if fits16 else pack3(arr.astype(np.int64))


def main():
    rng = np.random.default_rng(SEED)
    results = {}

    for domain, (tk, enc_name) in NATIVE.items():
        enc = tiktoken.get_encoding(enc_name)
        vocab = enc.n_vocab
        fits16 = vocab <= 0xFFFF
        train = load_ids(f"{domain}_train")
        test = load_ids(f"{domain}_test")
        r50k = tiktoken.get_encoding("r50k_base")

        # re-encode the shared corpus into THIS tokenizer, as elsewhere in the repo
        src = make_chunks(train, CHUNK * 4, 400, rng)
        train_ids = np.concatenate(
            [np.asarray(enc.encode(r50k.decode(w.tolist()), disallowed_special=()), dtype=np.int64)
             for w in src[:300]]
        )
        rank_of, token_of_rank = build_rank_table(train_ids, vocab)

        test_src = make_chunks(test, CHUNK * 4, N_CHUNKS + 30, rng)
        chunks = []
        for w in test_src:
            ids = enc.encode(r50k.decode(w.tolist()), disallowed_special=())
            if len(ids) >= CHUNK:
                chunks.append(np.asarray(ids[:CHUNK], dtype=np.int64))
            if len(chunks) >= N_CHUNKS:
                break

        freqs, ans_coder = build_ans_model(train_ids, vocab), None
        variants = {}

        def record(name, fn, dec=None):
            """dec(blob, n) -> token ids. Every variant is round-trip asserted:
            a ratio for an encoding that cannot be decoded back to the exact
            token IDs is not a result."""
            variants.setdefault(name, {"ratio": [], "enc_us": [], "dec_us": []})
            for ids in chunks:
                raw_len = len(enc.decode(ids.tolist()).encode("utf-8"))
                blob = fn(ids)
                if dec is not None:
                    assert np.array_equal(np.asarray(dec(blob, len(ids))), ids), name
                variants[name]["ratio"].append(raw_len / len(blob))
                variants[name]["enc_us"].append(timed_reps(lambda i=ids: fn(i)))
                if dec is not None:
                    variants[name]["dec_us"].append(timed_reps(lambda b=blob, n=len(ids): dec(b, n)))

        unpack = (lambda b, n: np.frombuffer(b, dtype="<u2")[:n].astype(np.int64)) if fits16 \
            else (lambda b, n: unpack3(b, n))

        # ---- baselines -----------------------------------------------------
        record("raw (fixed pack)", lambda i: fixed_pack(i, fits16), unpack)
        record("+lz4 (raw -> LZ4)", lambda i: lz4f.compress(fixed_pack(i, fits16)),
               lambda b, n: unpack(lz4f.decompress(b), n))
        record("+freq (svb)", lambda i: svb_encode_arr(rank_of[i]),
               lambda b, n: token_of_rank[svb_decode_arr(b, n)])

        # ---- hybrids -------------------------------------------------------
        # keeps byte alignment; small ranks give zero high bytes for LZ4 to eat
        record("+freq fixed -> LZ4", lambda i: lz4f.compress(fixed_pack(rank_of[i], fits16)),
               lambda b, n: token_of_rank[unpack(lz4f.decompress(b), n)])
        # compact first, then LZ4: shortest input, worst alignment
        record("+freq svb -> LZ4", lambda i: lz4f.compress(svb_encode_arr(rank_of[i])),
               lambda b, n: token_of_rank[svb_decode_arr(lz4f.decompress(b), n)])
        record("+freq leb128 -> LZ4", lambda i: lz4f.compress(leb128_encode(rank_of[i])),
               lambda b, n: token_of_rank[leb128_decode(lz4f.decompress(b))[:n]])
        record("+freq fixed -> LZ4-HC",
               lambda i: lz4f.compress(fixed_pack(rank_of[i], fits16), compression_level=12),
               lambda b, n: token_of_rank[unpack(lz4f.decompress(b), n)])

        results[domain] = {
            k: {
                "ratio": bootstrap_ci(np.array(v["ratio"]), rng),
                "encode_us": bootstrap_ci(np.array(v["enc_us"]), rng),
                "decode_us": bootstrap_ci(np.array(v["dec_us"]), rng) if v["dec_us"] else None,
            }
            for k, v in variants.items()
        }

        print(f"== {domain.upper()}  ({tk}, vocab {vocab}, {'uint16' if fits16 else '3-byte'})",
              flush=True)
        print(f"   {'variant':<24}{'ratio':>8}{'encode us':>11}{'decode us':>11}", flush=True)
        for k, v in results[domain].items():
            d = f"{v['decode_us'][0]:>11.1f}" if v["decode_us"] else f"{'-':>11}"
            print(f"   {k:<24}{v['ratio'][0]:>8.2f}{v['encode_us'][0]:>11.1f}{d}", flush=True)
        print(flush=True)

    OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
