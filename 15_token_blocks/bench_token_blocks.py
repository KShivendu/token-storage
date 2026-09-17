"""Token-native methods over ES/Lucene-style 16KB blocks.

The gap this fills: `07_kalcher_baseline/block_codecs_results.json` measures
blocks for BYTE codecs only (LZ4, zstd-19, brotli), and the 2x2 read-latency
table has a single `token/block` read number with no ratio and no encode cost.
So the talk compares per-document token storage against blocked byte storage,
which is not the comparison a real engine presents.

Design: documents are grouped into blocks EXACTLY as Lucene does and exactly as
07 does -- corpus-adjacent order, flush at 16 KB of raw UTF-8 or 128 docs,
trailing partial block still compressed. Both stores therefore hold the SAME
documents per block, so read amplification is identical and the ratios are
comparable. The only thing that changes is what gets written: the block's UTF-8
bytes, or the block's concatenated token IDs.

Reported per method:
  ratio        raw UTF-8 bytes of the block / compressed block bytes
  encode/doc   amortized -- cost of compressing a whole block / docs in it

Decode is NOT measured here. For the read comparison at block level, see
07_kalcher_baseline/read_latency_2x2_results.json, which has token/block at
39.8us against byte/block at 304.8us on prose.

Run:  uv run python 15_token_blocks/bench_token_blocks.py
"""

import gzip
import json
import sys
from pathlib import Path

import constriction
import lz4.frame as lz4f
import numpy as np
import tiktoken
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tnbench import (  # noqa: E402
    bootstrap_ci,
    build_ans_model,
    build_rank_table,
    load_ids,
    make_chunks,
    pack3,
    svb_encode_arr,
    timed_reps,
)

DICT_SIZE = 112 * 1024
BLOCK_BYTES = 16 * 1024
BLOCK_MAX_DOCS = 128
CHUNK = 512
N_DOCS = 300
SEED = 9012
NATIVE = {"prose": "r50k_base", "code": "cl100k_base", "hindi": "o200k_base"}
OUT = Path(__file__).resolve().parent / "results.json"


def token_id_dict(train_ids_per_window, level=22):
    """+dict as 03_latency defines it: a 112K zstd dictionary trained on RAW
    PACKED TOKEN-ID BYTES, not on UTF-8. tnbench.full_train_zstd_dict trains on
    text and is the byte-side `zstd --train`, a different codec."""
    zd = zstd.ZstdCompressionDict(zstd.train_dictionary(DICT_SIZE, train_ids_per_window).as_bytes())
    return zstd.ZstdCompressor(level=level, dict_data=zd)


def group_blocks(raw_byte_lists):
    """Lucene's CompressingStoredFieldsWriter batching, same as 07."""
    blocks, cur, cur_bytes = [], [], 0
    for raw in raw_byte_lists:
        if cur and (cur_bytes + len(raw) > BLOCK_BYTES or len(cur) >= BLOCK_MAX_DOCS):
            blocks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(raw)
        cur_bytes += len(raw)
    if cur:
        blocks.append(cur)  # trailing "dirty" block is still compressed
    return blocks


def main():
    rng = np.random.default_rng(SEED)
    r50k = tiktoken.get_encoding("r50k_base")
    zc22 = zstd.ZstdCompressor(level=22)
    zstd19 = zstd.ZstdCompressor(level=19)
    results = {}

    for domain, enc_name in NATIVE.items():
        enc = tiktoken.get_encoding(enc_name)
        vocab = enc.n_vocab
        fits16 = vocab <= 0xFFFF

        test_r50k = load_ids(f"{domain}_test")
        sampled = sorted(make_chunks(test_r50k, CHUNK, N_DOCS, rng), key=lambda c: c[0])
        texts = [r50k.decode(c.tolist()) for c in sampled]
        raws = [t.encode("utf-8") for t in texts]
        # each document's IDs in the corpus's own native tokenizer
        doc_ids = [np.asarray(enc.encode(t, disallowed_special=()), dtype=np.int64) for t in texts]

        # models trained on the TRAIN split only, never the documents above
        train_r50k = load_ids(f"{domain}_train")
        train_ids = np.concatenate(
            [
                np.asarray(enc.encode(r50k.decode(w.tolist()), disallowed_special=()), dtype=np.int64)
                for w in make_chunks(train_r50k, CHUNK, 400, rng)
            ]
        )
        rank_of, token_of_rank = build_rank_table(train_ids, vocab)
        ans_model = build_ans_model(train_ids, vocab)

        def _pack(a):
            return a.astype("<u2").tobytes() if fits16 else pack3(a.astype(np.int64))

        n_win = len(train_ids) // CHUNK
        zc_dict = token_id_dict(
            [_pack(train_ids[i * CHUNK : (i + 1) * CHUNK]) for i in range(n_win)]
        )

        idx = {id(r): i for i, r in enumerate(raws)}
        blocks = group_blocks(raws)

        pack = _pack

        def ans(a):
            c = constriction.stream.stack.AnsCoder()
            c.encode_reverse(a.astype(np.int32), ans_model)
            return c.get_compressed().tobytes()

        methods = {
            # byte side, for reference -- should reproduce 07
            "byte LZ4": ("byte", lambda b: lz4f.compress(b)),
            "byte gzip-9": ("byte", lambda b: gzip.compress(b, 9)),
            "byte zstd-19": ("byte", lambda b: zstd19.compress(b)),
            # token side, the gap this fills
            "tok raw": ("token", lambda a: pack(a)),
            "tok +freq": ("token", lambda a: svb_encode_arr(rank_of[a])),
            "tok +ANS": ("token", lambda a: ans(a)),
            "tok +freq->LZ4": ("token", lambda a: lz4f.compress(svb_encode_arr(rank_of[a]))),
            "tok raw->LZ4": ("token", lambda a: lz4f.compress(pack(a))),
            "tok +dict": ("token", lambda a: zc_dict.compress(pack(a))),
            "tok +zstd22": ("token", lambda a: zc22.compress(pack(a))),
        }

        per = {k: {"ratio": [], "enc": []} for k in methods}
        docs_per_block = []

        for block in blocks:
            raw_cat = b"".join(block)
            ids_cat = np.concatenate([doc_ids[idx[id(r)]] for r in block])
            docs_per_block.append(len(block))
            for name, (kind, fn) in methods.items():
                payload = raw_cat if kind == "byte" else ids_cat
                blob = fn(payload)
                per[name]["ratio"].append(len(raw_cat) / len(blob))
                per[name]["enc"].append(timed_reps(lambda p=payload, f=fn: f(p)) / len(block))

        results[domain] = {
            "median_docs_per_block": float(np.median(docs_per_block)),
            "n_blocks": len(blocks),
            "methods": {
                k: {
                    "ratio": bootstrap_ci(np.array(v["ratio"]), rng),
                    "encode_per_doc_us": bootstrap_ci(np.array(v["enc"]), rng),
                }
                for k, v in per.items()
            },
        }

        r = results[domain]
        print(f"== {domain.upper()}  {r['n_blocks']} blocks, "
              f"median {r['median_docs_per_block']:.0f} docs/block", flush=True)
        print(f"   {'method':<18}{'block ratio':>13}{'enc/doc us':>12}", flush=True)
        for k, v in r["methods"].items():
            print(f"   {k:<18}{v['ratio'][0]:>13.2f}{v['encode_per_doc_us'][0]:>12.1f}", flush=True)
        print(flush=True)

    OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
