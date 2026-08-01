"""
Re-runs the "does this generalize beyond OpenAI tokenizers" check on the
English (C4) corpus instead of the old, now-retired WikiText-103 numbers.
Same chunking methodology as the rest of the post: 512-token chunks, 40 test
chunks, RNG seed 3344, ANS table trained on the train split only.

For each tokenizer: uint32/16 raw packing ratio, 3-byte raw packing ratio,
+static ANS ratio, vs raw UTF-8 bytes.
"""
import os
import sys
import numpy as np
import tiktoken
from transformers import AutoTokenizer
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tnbench import make_chunks as _make_chunks, pack3, build_ans_model, load_ids

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

print(f"{'Tokenizer':<14}{'Vocab':>10}{'uint32/16':>12}{'3-byte':>10}{'+ANS':>10}")
for name, tok in TOKENIZERS.items():
    vocab_size = tok.vocab_size
    fits_uint16 = vocab_size <= 65536

    train_ids = tok.encode(train_text)
    ids_arr_train = np.array(train_ids, dtype=np.int64)
    ids_arr_train = ids_arr_train[(ids_arr_train >= 0) & (ids_arr_train < vocab_size)]
    model = build_ans_model(ids_arr_train, vocab_size)

    narrow_bytes, wide_bytes, ans_bytes, raw_bytes = [], [], [], []
    for text, rawlen in zip(texts, raw_byte_lens):
        ids = tok.encode(text)
        ids_arr = np.array(ids, dtype=np.int64)
        raw_bytes.append(rawlen)
        narrow_bytes.append(len(ids) * (2 if fits_uint16 else 4))
        wide_bytes.append(len(pack3(ids_arr)))

        c = constriction.stream.stack.AnsCoder()
        ids32 = np.clip(ids_arr, 0, vocab_size - 1).astype(np.int32)
        c.encode_reverse(ids32, model)
        ans_bytes.append(len(c.get_compressed().tobytes()))

    raw_bytes = np.array(raw_bytes)
    narrow_ratio = np.median(raw_bytes / np.array(narrow_bytes))
    wide_ratio = np.median(raw_bytes / np.array(wide_bytes))
    ans_ratio = np.median(raw_bytes / np.array(ans_bytes))
    tag = "(uint16)" if fits_uint16 else ""
    print(f"{name:<14}{vocab_size:>10}{narrow_ratio:>9.2f}x{tag:<3}{wide_ratio:>9.2f}x  {ans_ratio:>7.2f}x")
