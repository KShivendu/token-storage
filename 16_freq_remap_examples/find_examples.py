"""Real tokens whose ID position moves furthest under +freq.

The talk claims "a token you use constantly can sit at ID 40,000, a rare one
sits at ID 12". This finds the actual tokens that make that true, so the slide
can name one instead of asserting the shape.

BPE assigns IDs in merge order, which correlates with frequency in the
tokenizer's TRAINING corpus but says nothing about yours. +freq reassigns IDs
by descending frequency in your data, so a varint spends one byte on the tokens
you actually use.

Prints, for each corpus in its native tokenizer:
  - frequent tokens stuck at high IDs  (what +freq promotes)
  - low-ID tokens that are rare here   (what +freq demotes)
  - the byte each one costs before and after

Run:  uv run python 16_freq_remap_examples/find_examples.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tnbench import build_rank_table, load_ids  # noqa: E402

NATIVE = {"prose": "r50k_base", "code": "cl100k_base", "hindi": "o200k_base"}
OUT = Path(__file__).resolve().parent / "results.json"


def show(tok_bytes):
    """Printable form of a token, with leading space made visible."""
    try:
        s = tok_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return repr(tok_bytes)
    return "'" + s.replace(" ", "␣") + "'"


def varint_bytes(v):
    """LEB128 width: what the ID actually costs on disk."""
    return 1 if v < 128 else 2 if v < 16384 else 3


def main():
    out = {}
    for domain, enc_name in NATIVE.items():
        enc = tiktoken.get_encoding(enc_name)
        vocab = enc.n_vocab
        r50k = tiktoken.get_encoding("r50k_base")
        src = load_ids(f"{domain}_train")
        ids = (
            src.astype(np.int64)
            if enc_name == "r50k_base"
            else np.asarray(
                enc.encode(r50k.decode(src[:3_000_000].tolist()), disallowed_special=()),
                dtype=np.int64,
            )
        )
        counts = np.bincount(ids, minlength=vocab)
        rank_of, _ = build_rank_table(ids, vocab)
        total = counts.sum()

        def row(i):
            i = int(i)
            return {
                "id": i,
                "rank": int(rank_of[i]),
                "count": int(counts[i]),
                "per_million": round(counts[i] / total * 1e6, 1),
                "token": show(enc.decode_single_token_bytes(i)),
                "bytes_before": varint_bytes(i),
                "bytes_after": varint_bytes(int(rank_of[i])),
            }

        # PROMOTED: rank by BYTES SAVED, not by raw id distance. A token that
        # moves 2,000 places but stays a 2-byte varint saves nothing; one that
        # crosses 16384 -> 128 goes from 3 bytes to 1 on every occurrence.
        seen = np.where(counts > 0)[0]
        saved = np.array([varint_bytes(int(i)) - varint_bytes(int(rank_of[i])) for i in seen])
        impact = saved * counts[seen]
        promoted = [row(i) for i in seen[np.argsort(-impact)][:12]]

        # DEMOTED: a low ID (so BPE thought it common) that is rare in this
        # corpus, and now costs MORE. Rank by bytes lost per occurrence, then by
        # how far it fell, so the list is not all zero-count byte fragments.
        low = np.arange(min(4000, vocab))
        lost = np.array([varint_bytes(int(rank_of[i])) - varint_bytes(int(i)) for i in low])
        demoted = [row(i) for i in low[np.lexsort((-(rank_of[low].astype(np.int64) - low), -lost))][:12]]

        out[domain] = {"tokenizer": enc_name, "promoted": promoted, "demoted": demoted}

        print(f"\n{'='*78}\n{domain.upper()}  ({enc_name})\n{'='*78}", flush=True)
        print("PROMOTED by +freq -- ranked by total bytes saved in this corpus", flush=True)
        print(f"  {'token':<16}{'BPE id':>8}{'->':^5}{'rank':>7}{'per 1M':>10}{'bytes':>9}", flush=True)
        for r in promoted:
            b = f"{r['bytes_before']}->{r['bytes_after']}"
            print(f"  {r['token']:<16}{r['id']:>8}{'->':^5}{r['rank']:>7}"
                  f"{r['per_million']:>10.1f}{b:>9}", flush=True)
        print("\nDEMOTED by +freq -- low BPE id, rare in this corpus", flush=True)
        print(f"  {'token':<16}{'BPE id':>8}{'->':^5}{'rank':>7}{'per 1M':>10}{'bytes':>9}", flush=True)
        for r in demoted:
            b = f"{r['bytes_before']}->{r['bytes_after']}"
            print(f"  {r['token']:<16}{r['id']:>8}{'->':^5}{r['rank']:>7}"
                  f"{r['per_million']:>10.1f}{b:>9}", flush=True)

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
