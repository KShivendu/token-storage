"""
tnbench -- shared harness + codec/remap/varint helpers for the token-storage
benchmark suite.

Every numbered directory (00_-10_) imports from here instead of re-defining the
same corpus loader / chunk sampler / bootstrap / timing / streamvbyte / LEB128 /
ANS helpers. The bodies are the exact ones the committed results.json files were
produced with, so importing them changes no measured number: the only RNG-
consuming helpers (`make_chunks`, `bootstrap_ci`) are byte-for-byte the proven
versions, and the model/table builders (`build_rank_table`, `build_ans_model`)
draw no RNG at all.

Timing conventions (named, used consistently across benches):
  - timed_reps(fn, reps)  WARM: median of `reps` back-to-back calls per chunk.
                          Robust sample for sub-/few-microsecond ops whose single
                          perf_counter shot is dominated by scheduler/GC noise.
  - timed_once(fn)        SERVING-COLD: one perf_counter shot. Used for
                          tokenize/detokenize (100s of us; a warm 30-rep loop
                          would understate the real cold-text serving cost).

Heavy/optional deps (pyfastpfor, constriction) are imported lazily inside the
functions that need them, so this module imports cleanly in a minimal env.
"""
import os
import time
import lzma

import numpy as np
import zstandard as zstd

# corpus lives at <repo>/data/corpus regardless of which bench imports this
CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "corpus")

# shared codec settings (Kalcher baseline + others)
zstd_c22 = zstd.ZstdCompressor(level=22)
zstd_d = zstd.ZstdDecompressor()
LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]


# ── corpus / sampling ────────────────────────────────────────────────────────
def load_ids(name):
    """Load a corpus token-id array (e.g. 'prose_train') as int64."""
    return np.load(os.path.join(CORPUS_DIR, f"{name}.npy")).astype(np.int64)


def make_chunks(test_arr, chunk_size, n_chunks, rng):
    """Sample up to n_chunks non-overlapping chunk_size-token windows, without
    replacement, at random chunk-aligned offsets. Consumes exactly one
    rng.choice draw -- keep call order fixed to reproduce chunk selection."""
    max_chunks = len(test_arr) // chunk_size
    n = min(n_chunks, max_chunks)
    starts = rng.choice(max_chunks, size=n, replace=False) * chunk_size
    return [test_arr[s : s + chunk_size] for s in starts]


def bootstrap_ci(values, rng, n_boot=2000, alpha=0.10):
    """Median + (1-alpha) bootstrap CI of the median. Consumes one rng.integers
    draw of shape (n_boot, len(values))."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(np.median(values)), (float(values[0]), float(values[0]))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_medians = np.median(values[idx], axis=1)
    lo, hi = np.percentile(boot_medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(values)), (float(lo), float(hi))


def seeded_rng(*key):
    """Per-cell RNG: np.random.default_rng([SEED, idx...]). Makes a grid cell's
    chunk selection reproducible and independent of run order."""
    return np.random.default_rng(list(key))


# ── timing ───────────────────────────────────────────────────────────────────
def timed_reps(fn, reps=30):
    """WARM: median of `reps` back-to-back calls, microseconds."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e6)
    return float(np.median(ts))


def timed_once(fn):
    """SERVING-COLD: single perf_counter shot, microseconds."""
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6


# ── raw token-id packing ─────────────────────────────────────────────────────
def pack3(ids):
    """3 bytes/id big-endian packing (vocabs that overflow uint16)."""
    n = len(ids)
    out = np.zeros(n * 3, dtype=np.uint8)
    out[0::3] = (ids >> 16) & 0xFF
    out[1::3] = (ids >> 8) & 0xFF
    out[2::3] = ids & 0xFF
    return out.tobytes()


def unpack3(buf, n):
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, 3).astype(np.int64)
    return (arr[:, 0] << 16) | (arr[:, 1] << 8) | arr[:, 2]


# ── LEB128 varint (Kalcher container), vectorized in numpy ───────────────────
# 7 value bits + 1 continuation bit per byte. Remapped ranks are 0..vocab-1, so
# every value fits in at most 3 bytes. Vectorized so Kalcher isn't strawmanned
# with a slow Python byte loop -- the honest cost is the general compressor.
def leb128_encode(values):
    v = np.asarray(values, dtype=np.uint32)
    assert v.max(initial=0) < (1 << 21), "value exceeds 3-byte LEB128 range"
    nbytes = np.ones(len(v), dtype=np.int64) + (v >= 128) + (v >= 16384)
    ends = np.cumsum(nbytes)
    total = int(ends[-1]) if len(v) else 0
    starts = ends - nbytes
    out = np.zeros(total, dtype=np.uint8)
    b0 = (v & 0x7F).astype(np.uint8) | np.where(nbytes > 1, 0x80, 0).astype(np.uint8)
    out[starts] = b0
    m2 = nbytes > 1
    b1 = ((v[m2] >> 7) & 0x7F).astype(np.uint8) | np.where(nbytes[m2] > 2, 0x80, 0).astype(np.uint8)
    out[starts[m2] + 1] = b1
    m3 = nbytes > 2
    b2 = ((v[m3] >> 14) & 0x7F).astype(np.uint8)
    out[starts[m3] + 2] = b2
    return out.tobytes()


def leb128_decode(buf):
    b = np.frombuffer(buf, dtype=np.uint8)
    if len(b) == 0:
        return np.zeros(0, dtype=np.uint32)
    is_last = (b & 0x80) == 0
    grp = np.empty(len(b), dtype=np.int64)
    grp[0] = 0
    np.cumsum(is_last[:-1], out=grp[1:])
    n = int(is_last.sum())
    starts = np.searchsorted(grp, np.arange(n))
    k = np.arange(len(b), dtype=np.int64) - starts[grp]
    vals = np.zeros(n, dtype=np.uint64)
    contrib = ((b & 0x7F).astype(np.uint64)) << (7 * k).astype(np.uint64)
    np.add.at(vals, grp, contrib)
    return vals.astype(np.uint32)


# ── streamvbyte (repo +freq container), lazy pyfastpfor ──────────────────────
_svb = None


def _svb_codec():
    global _svb
    if _svb is None:
        import pyfastpfor
        _svb = pyfastpfor.getCodec("streamvbyte")
    return _svb


def svb_encode_arr(arr):
    codec = _svb_codec()
    arr = arr.astype(np.uint32)
    out = np.zeros(len(arr) * 2 + 1024, dtype=np.uint32)
    n_out = codec.encodeArray(arr, len(arr), out, len(out))
    return out[:n_out].tobytes()


def svb_decode_arr(payload, n):
    codec = _svb_codec()
    packed = np.frombuffer(payload, dtype=np.uint32)
    out = np.zeros(n + 1024, dtype=np.uint32)
    codec.decodeArray(packed, len(packed), out, n)
    return out[:n]


# ── frequency-remap + static ANS model (train-split tables, no RNG) ──────────
def build_rank_table(ids, vocab):
    """Descending-frequency rank table over `ids` (shared by +freq and Kalcher).
    Returns (rank_of[token_id] -> rank uint32, token_of_rank[rank] -> token_id
    int64). Deterministic: no RNG."""
    counts = np.bincount(np.asarray(ids, dtype=np.int64), minlength=vocab)
    order = np.argsort(-counts)
    rank_of = np.empty(vocab, dtype=np.uint32)
    rank_of[order] = np.arange(vocab, dtype=np.uint32)
    return rank_of, order.astype(np.int64)


def build_ans_model(ids, vocab):
    """Static Laplace(+1) unigram ANS model over `ids`. Caller supplies the id
    slice (e.g. first 400*512 train tokens, or the full train split) already
    range-filtered/clipped for HF tokenizers. Deterministic: no RNG."""
    import constriction
    counts = np.ones(vocab, dtype=np.int64)
    counts += np.bincount(np.asarray(ids, dtype=np.int64), minlength=vocab)
    probs = counts.astype(np.float64) / counts.sum()
    return constriction.stream.model.Categorical(probs, perfect=False)
