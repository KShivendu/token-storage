"""
Scaled (~1GB-text-per-domain) variant of prep_multidomain.py, for the
compression-ratio robustness check at enwik9 parity (Kalcher used 1GB).

Same three domains, same sources, same r50k storage as the original so the
existing ratio pipeline (r50k-decode -> byte codecs / o200k re-encode -> token
methods) is byte-for-byte comparable. The ONLY change is scale + a memory-safe
writer: the original accumulates a Python int list then np.array()s it, which
would need many GB of RAM at 200M+ tokens. Here we stream token ids into a small
buffer, flush the buffer's bytes to a raw .bin as we go (peak RAM ~ a few hundred
MB regardless of corpus size), then wrap the .bin into a proper uint32 .npy with
an O(1)-memory header write. This also lets the finite Hindi Wikipedia be taken
in FULL (test_target=None) without knowing its size in advance.

Targets (approx 1GB text/domain, per the byte/token ratios in the task brief):
  prose (English/C4):        train 30M,  test 220M tokens  (~1GB @ ~4.5 B/tok)
  code  (Python/codeparrot): train 30M,  test 420M tokens  (~1GB @ ~2.4 B/tok)
  hindi (wikimedia 20231101):train 20M,  test = FULL rest  (finite, report size)

Usage:
  uv run python 00_corpus_prep/prep_multidomain_1gb.py \
      [--domains prose,code,hindi] [--out data/corpus_1gb]
"""
import argparse
import os
import shutil
import time

import numpy as np
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("r50k_base")

# (train_target, test_target) in tokens. test_target=None => take all remaining.
TARGETS = {
    "prose": (30_000_000, 220_000_000),
    "code": (30_000_000, 420_000_000),
    "hindi": (20_000_000, None),
}
SOURCES = {
    "prose": (dict(path="allenai/c4", name="en", split="train", streaming=True), "text"),
    "code": (dict(path="codeparrot/codeparrot-clean", split="train", streaming=True), "content"),
    "hindi": (dict(path="wikimedia/wikipedia", name="20231101.hi", split="train", streaming=True), "text"),
}

FLUSH_TOKENS = 4_000_000  # buffer size before flushing to disk (~16 MB uint32)


class ShardWriter:
    """Append-only uint32 sink: buffers ids, flushes bytes to a .bin, then wraps
    into a .npy with a streaming header write (never loads the whole array)."""

    def __init__(self, bin_path):
        self.bin_path = bin_path
        self.f = open(bin_path, "wb")
        self.buf = []
        self.buf_n = 0
        self.count = 0

    def add(self, ids):
        self.buf.extend(ids)
        self.buf_n += len(ids)
        if self.buf_n >= FLUSH_TOKENS:
            self.flush()

    def flush(self):
        if self.buf_n:
            self.f.write(np.asarray(self.buf, dtype=np.uint32).tobytes())
            self.count += self.buf_n
            self.buf = []
            self.buf_n = 0

    def close_to_npy(self, npy_path):
        self.flush()
        self.f.close()
        with open(npy_path, "wb") as out:
            np.lib.format.write_array_header_2_0(
                out, {"descr": "<u4", "fortran_order": False, "shape": (self.count,)}
            )
            with open(self.bin_path, "rb") as g:
                shutil.copyfileobj(g, out, length=32 * 1024 * 1024)
        os.remove(self.bin_path)
        return self.count


def build_domain(name, out_dir):
    (ds_args, key) = SOURCES[name]
    train_target, test_target = TARGETS[name]
    print(f"\n=== {name}: {ds_args['path']} ===", flush=True)
    ds = load_dataset(**ds_args)

    tw = ShardWriter(os.path.join(out_dir, f"{name}_train.bin"))
    ew = ShardWriter(os.path.join(out_dir, f"{name}_test.bin"))
    train_docs = test_docs = 0
    t0 = time.perf_counter()
    last = t0

    for ex in ds:
        text = ex.get(key)
        if not text or len(text) < 200:
            continue
        ids = enc.encode(text, disallowed_special=())
        if not ids:
            continue
        if tw.count + tw.buf_n < train_target:
            tw.add(ids)
            train_docs += 1
        elif test_target is None or ew.count + ew.buf_n < test_target:
            ew.add(ids)
            test_docs += 1
        else:
            break
        now = time.perf_counter()
        if now - last > 30:
            tn = tw.count + tw.buf_n
            en = ew.count + ew.buf_n
            rate = (tn + en) / (now - t0) / 1e6
            print(
                f"  [{now - t0:6.0f}s] train={tn/1e6:6.1f}M test={en/1e6:6.1f}M "
                f"tok  ({rate:.2f} M tok/s)",
                flush=True,
            )
            last = now

    tn = tw.close_to_npy(os.path.join(out_dir, f"{name}_train.npy"))
    en = ew.close_to_npy(os.path.join(out_dir, f"{name}_test.npy"))
    dt = time.perf_counter() - t0
    # byte size of the r50k-decoded UTF-8 text for the TEST split (sampled est.)
    print(
        f"  DONE {name}: train={tn:,} tok ({train_docs} docs), "
        f"test={en:,} tok ({test_docs} docs) in {dt:.0f}s",
        flush=True,
    )
    return name, tn, en


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="prose,code,hindi")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "corpus_1gb"))
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    print("Output dir:", out_dir)
    summary = []
    for d in [x.strip() for x in args.domains.split(",") if x.strip()]:
        summary.append(build_domain(d, out_dir))
    print("\n=== SUMMARY (tokens) ===")
    for name, tn, en in summary:
        print(f"  {name:6s} train={tn:>13,}  test={en:>13,}")


if __name__ == "__main__":
    main()
