"""
Re-runs the "does this generalize beyond OpenAI tokenizers" check on the
English (C4) corpus instead of the old, now-retired WikiText-103 numbers.
Same chunking methodology as the rest of the post: 512-token chunks, 40 test
chunks, RNG seed 3344, ANS table trained on the train split only.

For each tokenizer: uint32/16 raw packing ratio, 3-byte raw packing ratio,
raw fixed-width token-ID packing at the CORRECT per-vocab byte width, and the
+static ANS ratio, vs raw UTF-8 bytes.

"raw (correct width)" is the fixed-width token-ID packing the paper reports as a
baseline: bytes/id is derived from each tokenizer's actual vocab size, never
hardcoded -- 2 bytes (uint16) for a vocab that fits in 16 bits (r50k), and 3
bytes (24-bit) for the large-vocab tokenizers (cl100k, o200k, Qwen2.5,
DeepSeek-V2, Gemma-2 all exceed uint16 but fit in 24 bits). 3 bytes, never 4
(uint32): the 3-byte packing is the paper's contribution. This is measured over
the exact same C4 English chunks / seed / methodology as the +ANS column, so the
raw band and the ANS band are directly comparable. The ratio has a bootstrap CI
(same bootstrap_ci helper the rest of the suite uses).
"""
import os
import sys
import numpy as np
import tiktoken
from transformers import AutoTokenizer
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import (
    make_chunks as _make_chunks,
    pack3,
    build_ans_model,
    load_ids,
    bootstrap_ci,
    seeded_rng,
)

CHUNK_SIZE = 512
N_CHUNKS = 40
RNG = np.random.default_rng(3344)

r50k = tiktoken.get_encoding("r50k_base")
train_ids_r50k = load_ids("prose_train")
test_ids_r50k = load_ids("prose_test")
# Same convention as bench_zstd_dict.py etc: a few hundred chunks is plenty
# for a frequency table. The full 8M-token train split re-tokenized with 3
# slow HF tokenizers took 10+ minutes and was still running — way overkill.
TRAIN_SAMPLE_TOKENS = 400 * CHUNK_SIZE
train_text = r50k.decode(train_ids_r50k[:TRAIN_SAMPLE_TOKENS].tolist())


def make_chunks(test_arr, chunk_size, n_chunks):
    return _make_chunks(test_arr, chunk_size, n_chunks, RNG)


chunks_r50k = make_chunks(test_ids_r50k, CHUNK_SIZE, N_CHUNKS)
texts = [r50k.decode(c.tolist()) for c in chunks_r50k]
raw_byte_lens = [len(t.encode("utf-8")) for t in texts]


class TiktokenWrap:
    def __init__(self, name):
        self.enc = tiktoken.get_encoding(name)
        self.vocab_size = {"r50k_base": 50257, "cl100k_base": 100277, "o200k_base": 200019}[name]

    def encode(self, text):
        return self.enc.encode(text, disallowed_special=())


class HFWrap:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.vocab_size = self.tok.vocab_size

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)


TOKENIZERS = {
    "r50k": TiktokenWrap("r50k_base"),
    "cl100k": TiktokenWrap("cl100k_base"),
    "o200k": TiktokenWrap("o200k_base"),
    "Qwen2.5": HFWrap("Qwen/Qwen2.5-7B"),
    "DeepSeek-V2": HFWrap("deepseek-ai/DeepSeek-V2-Lite"),
    "Gemma-2": HFWrap("google/gemma-2-9b"),
}

def bytes_per_id_for(vocab_size):
    """Fixed-width bytes/id from the actual vocab size (never hardcoded):
    2 bytes if the vocab fits in uint16, else 3 bytes (24-bit). Large-vocab
    tokenizers pack in 3 bytes, never 4."""
    if vocab_size <= 1 << 16:
        return 2
    if vocab_size <= 1 << 24:
        return 3
    raise ValueError(f"vocab {vocab_size} exceeds 24-bit raw packing")


print(f"{'Tokenizer':<14}{'Vocab':>10}{'b/id':>6}{'uint32/16':>12}{'3-byte':>9}{'raw(correct)':>26}{'+ANS':>9}")
raw_ratios_summary = {}
for idx, (name, tok) in enumerate(TOKENIZERS.items()):
    vocab_size = tok.vocab_size
    fits_uint16 = vocab_size <= 65536
    bytes_per_id = bytes_per_id_for(vocab_size)

    train_ids = tok.encode(train_text)
    ids_arr_train = np.array(train_ids, dtype=np.int64)
    ids_arr_train = ids_arr_train[(ids_arr_train >= 0) & (ids_arr_train < vocab_size)]
    model = build_ans_model(ids_arr_train, vocab_size)

    narrow_bytes, wide_bytes, ans_bytes, raw_bytes = [], [], [], []
    raw_correct_bytes = []
    for text, rawlen in zip(texts, raw_byte_lens):
        ids = tok.encode(text)
        ids_arr = np.array(ids, dtype=np.int64)
        raw_bytes.append(rawlen)
        narrow_bytes.append(len(ids) * (2 if fits_uint16 else 4))
        wide_bytes.append(len(pack3(ids_arr)))
        # raw fixed-width packing at the CORRECT per-vocab byte width
        raw_correct_bytes.append(len(ids) * bytes_per_id)

        c = constriction.stream.stack.AnsCoder()
        ids32 = np.clip(ids_arr, 0, vocab_size - 1).astype(np.int32)
        c.encode_reverse(ids32, model)
        ans_bytes.append(len(c.get_compressed().tobytes()))

    raw_bytes = np.array(raw_bytes)
    narrow_ratio = np.median(raw_bytes / np.array(narrow_bytes))
    wide_ratio = np.median(raw_bytes / np.array(wide_bytes))
    ans_ratio = np.median(raw_bytes / np.array(ans_bytes))
    # raw (correct width) with a bootstrap CI of the median, using the shared
    # helper. Per-tokenizer seeded RNG so the CI is reproducible and does not
    # perturb the chunk-selection RNG (chunks were drawn once, before the loop).
    raw_per_chunk = raw_bytes / np.array(raw_correct_bytes)
    raw_med, (raw_lo, raw_hi) = bootstrap_ci(raw_per_chunk, seeded_rng(3344, idx))
    raw_ratios_summary[name] = (vocab_size, bytes_per_id, raw_med, raw_lo, raw_hi, ans_ratio)
    tag = "(uint16)" if fits_uint16 else ""
    raw_str = f"{raw_med:.2f}x [{raw_lo:.2f},{raw_hi:.2f}]"
    print(
        f"{name:<14}{vocab_size:>10}{bytes_per_id:>6}{narrow_ratio:>9.2f}x{tag:<3}"
        f"{wide_ratio:>8.2f}x{raw_str:>26}{ans_ratio:>8.2f}x"
    )

raw_meds = [v[2] for v in raw_ratios_summary.values()]
ans_meds = [v[5] for v in raw_ratios_summary.values()]
print(
    f"\nraw fixed-width band across the six tokenizers: "
    f"{min(raw_meds):.2f}x - {max(raw_meds):.2f}x  "
    f"(for reference, +ANS band here: {min(ans_meds):.2f}x - {max(ans_meds):.2f}x)"
)
