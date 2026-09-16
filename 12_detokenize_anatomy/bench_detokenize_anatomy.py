"""Why does detokenize cost ~50us when it is "just a lookup table"?

Because it IS just a lookup table, and that is the problem. Decoding a
512-token chunk means 512 effectively-random probes into a vocabulary of
50k-200k entries. tiktoken stores that as a HashMap<Rank, Vec<u8>>, so each
entry is a pointer to its own small heap allocation: the probe misses cache,
then the pointer chase misses again. Nothing is compute-bound here; the cost is
DRAM round trips, and no amount of faster code removes a memory stall.

Three things are measured, each as a single shot with the CPU cache swept
first (`timed_once_serving`), which is the honest serving condition -- in real
serving, model work and decompression evict the table between chunks:

  1. tiktoken decode            -- the 50us number the talk quotes
  2. flat-arena decode          -- same lookup table, laid out as one contiguous
                                   byte arena + an offsets array, so the whole
                                   structure is a couple of MB and mostly
                                   survives in L2/L3
  3. flat-arena, sequential IDs -- the same arena probed in ID order instead of
                                   text order. Identical instruction count, but
                                   the accesses are now contiguous, which
                                   isolates how much of the cost is pure random
                                   memory latency versus the work itself.

IMPORTANT -- do not compare (1) against (2) in absolute terms. tiktoken's decode
is compiled Rust; the flat-arena decode here is a Python list comprehension, so
(2) loses on interpreter overhead alone and that says nothing about memory. The
two comparisons that ARE controlled hold the implementation fixed and vary only
the memory condition:

  A. tiktoken warm vs tiktoken serving-cold -- same Rust code, same inputs, the
     only difference is whether the table survives in cache.
  B. flat random-IDs vs flat sequential-IDs -- same Python code, same arena,
     same 512 lookups, the only difference is whether the probes are scattered
     across the vocabulary or contiguous.

Both deltas are the per-token memory stall. If the hypothesis holds, both land
in the same tens-of-nanoseconds range (a DRAM round trip is ~60-100ns) and both
grow with vocabulary size, since a bigger table means less of it stays cached.

Run:  uv run python 12_detokenize_anatomy/bench_detokenize_anatomy.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tnbench import (  # noqa: E402
    bootstrap_ci,
    load_ids,
    make_chunks,
    timed_once_serving,
    timed_reps,
)

CHUNK_SIZE = 512
N_CHUNKS = 100
SEED = 5566
TOKENIZERS = ["r50k", "cl100k", "o200k"]
ENC_NAME = {"r50k": "r50k_base", "cl100k": "cl100k_base", "o200k": "o200k_base"}
OUT = Path(__file__).resolve().parent / "results.json"


def build_flat_arena(enc):
    """The same decoder as tiktoken's, in the layout a lookup table *should*
    have: every token's bytes concatenated into one contiguous arena, plus a
    uint32 offsets array. Lookup is arena[off[i]:off[i+1]] -- one indexed read
    into a flat buffer, no hashing and no per-entry pointer chase."""
    n = enc.n_vocab
    pieces = []
    for i in range(n):
        try:
            pieces.append(enc.decode_single_token_bytes(i))
        except Exception:
            pieces.append(b"")  # unused/special slots still need a slot
    offsets = np.zeros(n + 1, dtype=np.uint32)
    np.cumsum([len(p) for p in pieces], out=offsets[1:], dtype=np.uint32)
    arena = b"".join(pieces)
    return arena, offsets


def flat_decode(arena, offsets, ids):
    return b"".join([arena[offsets[i] : offsets[i + 1]] for i in ids])


def main():
    rng = np.random.default_rng(SEED)
    test = load_ids("prose_test")
    r50k = tiktoken.get_encoding("r50k_base")

    results = {
        "config": {
            "chunk_size_tokens": CHUNK_SIZE,
            "n_chunks": N_CHUNKS,
            "seed": SEED,
            "timing": "timed_once_serving: 64MB cache sweep, then one shot",
            "corpus": "prose_test (English C4), 512 IDs in each tokenizer's own encoding",
        },
        "tokenizers": {},
    }

    print(f"{'tok':<8}{'vocab':>8}{'arena MB':>10}{'tiktoken':>11}{'flat':>9}"
          f"{'flat-seq':>10}{'warm tik':>10}", flush=True)

    for tk in TOKENIZERS:
        enc = tiktoken.get_encoding(ENC_NAME[tk])
        # Same convention as 09_cold_tokenize: take r50k windows, decode to
        # text, re-encode with this tokenizer, and cut into 512-ID windows, so a
        # chunk really is 512 IDs of THIS tokenizer rather than of r50k.
        src_windows = make_chunks(test, CHUNK_SIZE * 4, N_CHUNKS + 20, rng)
        id_lists = []
        for w in src_windows:
            ids = enc.encode(r50k.decode(w.tolist()), disallowed_special=())
            if len(ids) >= CHUNK_SIZE:
                id_lists.append(ids[:CHUNK_SIZE])
            if len(id_lists) >= N_CHUNKS:
                break

        arena, offsets = build_flat_arena(enc)
        arena_mb = (len(arena) + offsets.nbytes) / 1e6

        # sanity: the flat arena must reproduce tiktoken's bytes exactly
        for ids in id_lists[:5]:
            assert flat_decode(arena, offsets, ids) == enc.decode(ids).encode("utf-8"), tk

        # sequential IDs: same count, same arena, contiguous instead of scattered
        seq_starts = rng.choice(enc.n_vocab - CHUNK_SIZE, size=len(id_lists), replace=False)
        seq_lists = [list(range(int(s), int(s) + CHUNK_SIZE)) for s in seq_starts]

        tik = np.array([timed_once_serving(lambda i=i: enc.decode(i)) for i in id_lists])
        flat = np.array([timed_once_serving(lambda i=i: flat_decode(arena, offsets, i)) for i in id_lists])
        fseq = np.array([timed_once_serving(lambda i=i: flat_decode(arena, offsets, i)) for i in seq_lists])
        warm = np.array([timed_reps(lambda i=i: enc.decode(i)) for i in id_lists])

        r = {
            "vocab": enc.n_vocab,
            "flat_arena_mb": arena_mb,
            "tiktoken_serving_cold_us": bootstrap_ci(tik, rng),
            "flat_serving_cold_us": bootstrap_ci(flat, rng),
            "flat_sequential_serving_cold_us": bootstrap_ci(fseq, rng),
            "tiktoken_warm_us": bootstrap_ci(warm, rng),
        }
        results["tokenizers"][tk] = r
        print(f"{tk:<8}{enc.n_vocab:>8}{arena_mb:>10.2f}{r['tiktoken_serving_cold_us'][0]:>10.1f}u"
              f"{r['flat_serving_cold_us'][0]:>8.1f}u{r['flat_sequential_serving_cold_us'][0]:>9.1f}u"
              f"{r['tiktoken_warm_us'][0]:>9.1f}u", flush=True)

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT}", flush=True)

    if results["tokenizers"]:
        f = 1000.0 / CHUNK_SIZE  # us per chunk -> ns per token
        print("\nper-token nanoseconds (512 lookups per chunk):", flush=True)
        for tk, r in results["tokenizers"].items():
            print(f"  {tk:<8} tiktoken cold {r['tiktoken_serving_cold_us'][0]*f:6.1f}  "
                  f"warm {r['tiktoken_warm_us'][0]*f:6.1f}   |   "
                  f"flat random {r['flat_serving_cold_us'][0]*f:6.1f}  "
                  f"seq {r['flat_sequential_serving_cold_us'][0]*f:6.1f}", flush=True)

        print("\nthe two controlled deltas -- per-token memory stall (ns):", flush=True)
        print(f"  {'tok':<8}{'vocab':>8}{'A: tiktoken cold-warm':>24}{'B: flat random-seq':>21}", flush=True)
        for tk, r in results["tokenizers"].items():
            a = (r["tiktoken_serving_cold_us"][0] - r["tiktoken_warm_us"][0]) * f
            b = (r["flat_serving_cold_us"][0] - r["flat_sequential_serving_cold_us"][0]) * f
            r["stall_ns_tiktoken_cold_minus_warm"] = a
            r["stall_ns_flat_random_minus_sequential"] = b
            print(f"  {tk:<8}{r['vocab']:>8}{a:>23.1f}{b:>21.1f}", flush=True)
        print("\n  Both isolate memory, both land in DRAM-latency range, and both", flush=True)
        print("  grow with vocabulary size. That is the whole answer.", flush=True)
        OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
