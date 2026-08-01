"""
Re-runs "how much does each step contribute: entropy coder vs tokenizer" on
English (C4), 512-token chunks, same seed/convention as the rest of the post
(WikiText is retired). Also adds streamvbyte-without-remap vs
freq-remap+streamvbyte, to isolate how much of +freq's ratio comes from the
rank-remap step itself vs the streamvbyte codec.
"""
import os
import sys
import numpy as np
import tiktoken
import zstandard as zstd
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import make_chunks as _make_chunks, svb_encode_arr, build_rank_table, load_ids

CHUNK_SIZE = 512
N_CHUNKS = 40
RNG = np.random.default_rng(3344)

r50k = tiktoken.get_encoding("r50k_base")
VOCAB = 50257

train_ids = load_ids("prose_train")
test_ids = load_ids("prose_test")
train_text = r50k.decode(train_ids[: 400 * CHUNK_SIZE].tolist())


def make_chunks(test_arr, chunk_size, n_chunks):
    return _make_chunks(test_arr, chunk_size, n_chunks, RNG)


chunks = make_chunks(test_ids, CHUNK_SIZE, N_CHUNKS)
texts = [r50k.decode(c.tolist()) for c in chunks]
raw_byte_lens = np.array([len(t.encode("utf-8")) for t in texts])

# ── ANS on raw UTF-8 bytes, no tokenizer (order-0 byte model) ────────────────
train_bytes = train_text.encode("utf-8")
byte_counts = np.bincount(np.frombuffer(train_bytes, dtype=np.uint8), minlength=256).astype(np.float64) + 1
byte_probs = byte_counts / byte_counts.sum()
byte_model = constriction.stream.model.Categorical(byte_probs, perfect=False)

ans_byte_sizes = []
for t in texts:
    raw = t.encode("utf-8")
    ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
    c = constriction.stream.stack.AnsCoder()
    c.encode_reverse(ids, byte_model)
    ans_byte_sizes.append(len(c.get_compressed().tobytes()))
ans_byte_ratio = np.median(raw_byte_lens / np.array(ans_byte_sizes))

# ── zstd (LZ77 + entropy) on packed r50k token-ID bytes (post-tokenization) ──
zc = zstd.ZstdCompressor(level=19)
zstd_on_ids_sizes = []
for c in chunks:
    packed = c.astype(np.uint16).tobytes()
    zstd_on_ids_sizes.append(len(zc.compress(packed)))
zstd_on_ids_ratio = np.median(raw_byte_lens / np.array(zstd_on_ids_sizes))

# ── token-level ANS (r50k), for reference (already known: 3.3x at 512 tok) ──
train_r50k_ids = r50k.encode(train_text)
tok_counts = np.ones(VOCAB, dtype=np.int64)
tok_counts += np.bincount(train_r50k_ids, minlength=VOCAB)
tok_probs = tok_counts.astype(np.float64) / tok_counts.sum()
tok_model = constriction.stream.model.Categorical(tok_probs, perfect=False)
ans_tok_sizes = []
for c in chunks:
    ids32 = c.astype(np.int32)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(ids32, tok_model)
    ans_tok_sizes.append(len(coder.get_compressed().tobytes()))
ans_tok_ratio = np.median(raw_byte_lens / np.array(ans_tok_sizes))

# ── streamvbyte WITHOUT rank-remap (plain BPE merge-order IDs) ───────────────
svb_noremap_sizes = []
for c in chunks:
    ids_u32 = c.astype(np.uint32)
    svb_noremap_sizes.append(len(svb_encode_arr(ids_u32)))
svb_noremap_ratio = np.median(raw_byte_lens / np.array(svb_noremap_sizes))

# ── streamvbyte WITH rank-remap (+freq, for reference: already known 2.62x) ──
rank_of, _ = build_rank_table(train_r50k_ids, VOCAB)
svb_remap_sizes = []
for c in chunks:
    remapped = rank_of[c.astype(np.int64)]
    svb_remap_sizes.append(len(svb_encode_arr(remapped)))
svb_remap_ratio = np.median(raw_byte_lens / np.array(svb_remap_sizes))

print("=== Entropy coder vs tokenizer, English (C4), 512-token chunks ===")
print(f"ANS on raw UTF-8 bytes (no tokenizer):        {ans_byte_ratio:.2f}x")
print(f"Token-level ANS (r50k):                       {ans_tok_ratio:.2f}x")
print(f"zstd-19 on packed r50k token-ID bytes:         {zstd_on_ids_ratio:.2f}x")
print()
print("=== streamvbyte alone vs +freq (rank-remap contribution) ===")
print(f"streamvbyte, no remap (BPE merge-order IDs):   {svb_noremap_ratio:.2f}x")
print(f"streamvbyte + rank-remap (+freq):               {svb_remap_ratio:.2f}x")
