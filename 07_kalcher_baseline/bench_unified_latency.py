"""
Task B: ONE unified English latency table, so the paper's latency table AND
figure can be regenerated from a single internally-consistent source.

Every method is measured with ONE definition of each direction:
  agent WRITE = start from token IDs, produce the stored bytes.
    byte codec (LZ4): detokenize(ids -> text) + compress(text bytes).
    token-native: pack / +freq-encode / ANS-encode / Kalcher-encode from IDs.
  agent READ = start from stored bytes, produce token IDs.
    byte codec (LZ4): decompress -> text -> tokenize -> IDs.
    token-native: unpack / +freq-decode(INCLUDING rank un-permute) /
                  ANS-decode / Kalcher-decode(INCLUDING rank un-permute).

The un-permute (rank -> token id lookup) is folded INTO the read for +freq and
Kalcher and timed as one function -- this is the single consistent accounting
the coordinator asked for (bench_kalcher.py and the paper's Table 2 measured it
differently, which is why their +freq read numbers disagreed).

English (C4) only, 512-token chunks, seed 3344, median-of-30-reps per chunk
then bootstrap over 40 chunks -- same robust methodology as
03_latency/bench_agent_mode_v2.py.
"""
import os
import sys
import json
import lzma
import numpy as np
import tiktoken
import lz4.frame as lz4f
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import (
    load_ids, make_chunks, bootstrap_ci, timed_reps, timed_once,
    leb128_encode, leb128_decode, svb_encode_arr, svb_decode_arr,
    build_rank_table, build_ans_model, zstd_c22, zstd_d, LZMA_FILTERS,
)

CHUNK_SIZE = 512
N_CHUNKS = 40
VOCAB = 50257
ANS_TRAIN_TOKENS = 400 * CHUNK_SIZE
SEED = 3344

r50k = tiktoken.get_encoding("r50k_base")


def main():
    rng = np.random.default_rng(SEED)
    train = load_ids("prose_train")
    test = load_ids("prose_test")

    rank_of, token_of_rank = build_rank_table(train, VOCAB)
    ans_model = build_ans_model(train[:ANS_TRAIN_TOKENS], VOCAB)

    chunks = make_chunks(test, CHUNK_SIZE, N_CHUNKS, rng)
    texts = [r50k.decode(c.tolist()) for c in chunks]

    methods = ["LZ4", "r50k raw", "r50k +freq", "r50k +ANS", "Kalcher(LZMA)", "Kalcher(zstd)"]
    write_t = {m: [] for m in methods}
    read_t = {m: [] for m in methods}

    for chunk, text in zip(chunks, texts):
        ids = chunk.astype(np.int64)
        n = len(ids)
        remapped = rank_of[ids]
        text_bytes = text.encode("utf-8")

        # ---- LZ4 (byte codec) ----
        # write = detokenize(IDs -> text) + compress ; read = decompress +
        # tokenize(text -> IDs). Codec halves timed median-of-30 (fast);
        # tokenize/detokenize timed single-shot (paper methodology), so the
        # mandatory ~hundreds-of-us tokenize cost isn't understated by warm
        # cache. Per-chunk latency = sum of the two components.
        lz4_payload = lz4f.compress(text_bytes)
        # LZ4 stores TEXT, so the artifact round-trips at the text level; the
        # agent recovers IDs by tokenizing that text (may differ from the
        # original chunk IDs only at chunk boundaries -- expected for a byte
        # codec, and exactly the tokenize cost we want to charge it).
        assert lz4f.decompress(lz4_payload) == text_bytes
        detok_us = timed_once(lambda i=ids: r50k.decode(i.tolist()))
        compress_us = timed_reps(lambda b=text_bytes: lz4f.compress(b))
        decompress_us = timed_reps(lambda p=lz4_payload: lz4f.decompress(p))
        tok_us = timed_once(lambda t=text: r50k.encode(t, disallowed_special=()))
        write_t["LZ4"].append(detok_us + compress_us)
        read_t["LZ4"].append(decompress_us + tok_us)

        # ---- r50k raw ----
        packed = ids.astype(np.uint16).tobytes()
        write_t["r50k raw"].append(timed_reps(lambda i=ids: i.astype(np.uint16).tobytes()))
        read_t["r50k raw"].append(timed_reps(lambda p=packed: np.frombuffer(p, dtype=np.uint16)))

        # ---- r50k +freq: write=remap+svbenc, read=svbdec+un-permute ----
        svb_payload = svb_encode_arr(remapped)

        def freq_write(i=ids):
            return svb_encode_arr(rank_of[i])

        def freq_read(p=svb_payload, n=n):
            return token_of_rank[svb_decode_arr(p, n)]

        assert np.array_equal(freq_read(), ids)
        write_t["r50k +freq"].append(timed_reps(freq_write))
        read_t["r50k +freq"].append(timed_reps(freq_read))

        # ---- r50k +ANS ----
        ids32 = ids.astype(np.int32)

        def ans_write(ids32=ids32):
            c = constriction.stream.stack.AnsCoder()
            c.encode_reverse(ids32, ans_model)
            return c.get_compressed().tobytes()

        ans_arr = np.frombuffer(ans_write(), dtype=np.uint32).copy()

        def ans_read(a=ans_arr, n=n):
            c2 = constriction.stream.stack.AnsCoder(a.copy())
            return c2.decode(ans_model, n)

        assert np.array_equal(np.asarray(ans_read()), ids32)
        write_t["r50k +ANS"].append(timed_reps(ans_write))
        read_t["r50k +ANS"].append(timed_reps(ans_read))

        # ---- Kalcher(LZMA / zstd): write=remap+leb128+compress, read=decompress+leb128+un-permute ----
        varint = leb128_encode(remapped)
        lzma_payload = lzma.compress(varint, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
        zstd_payload = zstd_c22.compress(varint)

        def klzma_write(i=ids):
            return lzma.compress(leb128_encode(rank_of[i]), format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)

        def kzstd_write(i=ids):
            return zstd_c22.compress(leb128_encode(rank_of[i]))

        def klzma_read(p=lzma_payload):
            v = lzma.decompress(p, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
            return token_of_rank[leb128_decode(v)]

        def kzstd_read(p=zstd_payload):
            v = zstd_d.decompress(p)
            return token_of_rank[leb128_decode(v)]

        assert np.array_equal(klzma_read(), ids)
        assert np.array_equal(kzstd_read(), ids)
        write_t["Kalcher(LZMA)"].append(timed_reps(klzma_write))
        read_t["Kalcher(LZMA)"].append(timed_reps(klzma_read))
        write_t["Kalcher(zstd)"].append(timed_reps(kzstd_write))
        read_t["Kalcher(zstd)"].append(timed_reps(kzstd_read))

    write_ci = {m: bootstrap_ci(write_t[m], rng) for m in methods}
    read_ci = {m: bootstrap_ci(read_t[m], rng) for m in methods}

    print(f"\n{'='*80}")
    print("UNIFIED English agent latency (median us [90% CI]), 512-token chunks, median-of-30-reps")
    print("write = from token IDs; read = to token IDs (incl. rank un-permute for +freq/Kalcher)")
    print(f"{'='*80}")
    print(f"  {'method':<16}{'agent write us':>26}{'agent read us':>26}")
    for m in methods:
        w, r = write_ci[m], read_ci[m]
        print(
            f"  {m:<16}"
            + f"{w[0]:.1f}[{w[1][0]:.1f},{w[1][1]:.1f}]".rjust(26)
            + f"{r[0]:.1f}[{r[1][0]:.1f},{r[1][1]:.1f}]".rjust(26)
        )

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing["taskB_english_unified_latency"] = {
        "config": {
            "domain": "prose (English C4)", "chunk_size": CHUNK_SIZE, "n_chunks": N_CHUNKS,
            "reps": 30, "seed": SEED,
            "write": "from token IDs; LZ4 = detokenize + compress",
            "read": "to token IDs; LZ4 = decompress + tokenize; +freq/Kalcher INCLUDE rank un-permute",
            "lzma": "FORMAT_RAW LZMA2 preset=9|EXTREME", "zstd": "level 22",
        },
        "write_us": {m: write_ci[m] for m in methods},
        "read_us": {m: read_ci[m] for m in methods},
    }
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print("\nMerged into results.json (taskB_english_unified_latency)")


if __name__ == "__main__":
    main()
