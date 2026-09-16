"""Sanity check: why does LZ4 only reach ~1.27x here when it is widely quoted at 2-2.5x?

The talk claims LZ4 barely compresses English prose (1.27x). Public LZ4
benchmarks routinely show 2-2.5x on text, so either the quoted figure measures
something different or this repo has a bug. This settles which.

Two hypotheses, both tested against the same corpus:

  H1  INPUT SIZE. LZ4 is LZ77: it compresses by pointing at earlier repeats
      inside its window. A 512-token chunk is ~2.3 KB, so there is very little
      earlier text to point at. Public numbers come from whole files (Silesia,
      enwik8) that are megabytes, where the same phrases recur thousands of
      times. Sweep one contiguous prose stream from 1 KB to 16 MB and watch the
      ratio climb.

  H2  FAST vs HC. `lz4.frame.compress` defaults to compression_level=0, which
      is LZ4 fast. LZ4-HC (level 9-12) searches much harder for matches and is
      what some published figures use. Run both at every size.

Run:  uv run python 13_lz4_sanity/bench_lz4_sanity.py
"""

import json
import sys
from pathlib import Path

import lz4.frame as lz4f
import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tnbench import load_ids  # noqa: E402

SIZES = [1 << k for k in range(10, 25)]  # 1 KB .. 16 MB
OUT = Path(__file__).resolve().parent / "results.json"


def main():
    r50k = tiktoken.get_encoding("r50k_base")
    ids = load_ids("prose_test")
    # one contiguous stream, so larger sizes are strictly supersets of smaller
    text = r50k.decode(ids[: 6_000_000].tolist())
    blob = text.encode("utf-8")
    print(f"corpus: {len(blob)/1e6:.1f} MB of English prose (C4)\n", flush=True)

    print(f"{'input':>10}{'LZ4 fast':>11}{'LZ4-HC':>10}{'gzip -9':>10}", flush=True)
    rows = []
    for n in SIZES:
        if n > len(blob):
            break
        raw = blob[:n]
        fast = len(raw) / len(lz4f.compress(raw))
        hc = len(raw) / len(lz4f.compress(raw, compression_level=12))
        import gzip

        gz = len(raw) / len(gzip.compress(raw, 9))
        rows.append({"bytes": n, "lz4_fast": fast, "lz4_hc": hc, "gzip9": gz})
        label = f"{n//1024} KB" if n < 1 << 20 else f"{n//(1<<20)} MB"
        print(f"{label:>10}{fast:>10.2f}x{hc:>9.2f}x{gz:>9.2f}x", flush=True)

    # The talk's actual condition: independent 512-token chunks, not one stream.
    chunk_ids = ids[: 512 * 200].reshape(200, 512)
    chunks = [r50k.decode(c.tolist()).encode("utf-8") for c in chunk_ids]
    per_chunk_fast = float(np.median([len(c) / len(lz4f.compress(c)) for c in chunks]))
    per_chunk_hc = float(
        np.median([len(c) / len(lz4f.compress(c, compression_level=12)) for c in chunks])
    )
    med_bytes = float(np.median([len(c) for c in chunks]))

    # Same bytes, concatenated into one buffer: isolates "small input" from
    # "this text is just not very compressible".
    joined = b"".join(chunks)
    joined_fast = len(joined) / len(lz4f.compress(joined))

    print(f"\n{'':-<44}", flush=True)
    print(f"the talk's condition: 200 independent 512-token chunks", flush=True)
    print(f"  median chunk size        {med_bytes:8.0f} bytes", flush=True)
    print(f"  LZ4 fast, per chunk      {per_chunk_fast:8.2f}x   <- the 1.27x in the deck", flush=True)
    print(f"  LZ4-HC,   per chunk      {per_chunk_hc:8.2f}x", flush=True)
    print(f"  same bytes, one buffer   {joined_fast:8.2f}x   ({len(joined)/1e6:.1f} MB)", flush=True)

    OUT.write_text(
        json.dumps(
            {
                "size_sweep": rows,
                "talk_condition": {
                    "n_chunks": len(chunks),
                    "median_chunk_bytes": med_bytes,
                    "lz4_fast_per_chunk": per_chunk_fast,
                    "lz4_hc_per_chunk": per_chunk_hc,
                    "lz4_fast_concatenated": joined_fast,
                },
            },
            indent=1,
        )
    )
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
