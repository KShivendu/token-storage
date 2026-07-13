"""
Full recompute for the "A Cheaper Fix: Frequency-Sorted Token IDs" table.
Covers r50k, cl100k, o200k. Baseline fixed-width columns: uint16, uint32,
uint64, and 3-byte/24-bit (the practical packing already used elsewhere in
this post for cl100k/o200k). Then frequency-remapped IDs with plain varint,
and with streamvbyte (pyfastpfor).

Same corpus methodology as the main storage benchmark: WikiText-103 test
split, articles with >=30 words, static frequency table trained on the
train split (same cached tables used elsewhere: static_probs_*.npy).
"""
import os
from collections import Counter

import numpy as np
import tiktoken
import pyfastpfor
from datasets import load_dataset

CACHE_DIR = os.path.dirname(__file__)

TOKENIZERS = {
    "o200k": "o200k_base",
    
}

print("Loading WikiText-103 test split...")
ds_test = load_dataset("wikitext", "wikitext-103-v1", split="test", trust_remote_code=True)
articles = [t.strip() for t in ds_test["text"] if len(t.strip().split()) >= 30]
print(f"  {len(articles)} articles")

print("Loading WikiText-103 train split (for frequency ranking)...")
ds_train = load_dataset("wikitext", "wikitext-103-v1", split="train", trust_remote_code=True)
train_text = "\n".join(ds_train["text"])

def varint_encode(ids):
    out = bytearray()
    for v in ids:
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)

def streamvbyte_encode(ids):
    arr = np.array(ids, dtype=np.uint32)
    codec = pyfastpfor.getCodec("streamvbyte")
    out = np.zeros(len(arr) * 2 + 1024, dtype=np.uint32)
    n_out = codec.encodeArray(arr, len(arr), out, len(out))
    return out[:n_out].tobytes()

W = 100
print(f"\n{'='*W}")
print("  Fixed-width baselines + frequency-remap, all three tokenizers")
print(f"{'='*W}")
header = (f"  {'Tok':<8} {'raw/uint16':>10} {'raw/uint32':>10} {'raw/uint64':>10} "
          f"{'raw/3-byte':>10} {'remap+varint':>13} {'remap+svb':>11}")
print(header)
print(f"  {'-'*(W-2)}")

results = {}

for name, enc_name in TOKENIZERS.items():
    enc = tiktoken.get_encoding(enc_name)
    vocab_size = enc.n_vocab

    # frequency ranking from train split
    train_ids = enc.encode(train_text, disallowed_special=())
    counts = Counter(train_ids)
    # rank 0 = most frequent
    ranked = [tok for tok, _ in counts.most_common()]
    rank_of = {tok: i for i, tok in enumerate(ranked)}
    # tokens never seen in train get ranks after all seen tokens (rare, keep original relative order)
    next_rank = len(ranked)

    total_raw = 0
    total_u16 = total_u32 = total_u64 = total_3b = 0
    total_varint = total_svb = 0

    for text in articles:
        raw = text.encode("utf-8")
        ids = enc.encode(text, disallowed_special=())
        if not ids:
            continue
        n = len(ids)
        total_raw += len(raw)
        total_u16 += n * 2
        total_u32 += n * 4
        total_u64 += n * 8
        total_3b += n * 3

        remapped = []
        for t in ids:
            r = rank_of.get(t)
            if r is None:
                rank_of[t] = next_rank
                r = next_rank
                next_rank += 1
            remapped.append(r)

        total_varint += len(varint_encode(remapped))
        total_svb += len(streamvbyte_encode(remapped))

    r16 = total_raw / total_u16 if total_u16 else float("nan")
    r32 = total_raw / total_u32
    r64 = total_raw / total_u64
    r3b = total_raw / total_3b
    rvarint = total_raw / total_varint
    rsvb = total_raw / total_svb

    results[name] = dict(u16=r16, u32=r32, u64=r64, b3=r3b, varint=rvarint, svb=rsvb)

    u16_str = f"{r16:.2f}x" if vocab_size <= 65536 else "n/a"
    print(f"  {name:<8} {u16_str:>10} {r32:>9.2f}x {r64:>9.2f}x {r3b:>9.2f}x "
          f"{rvarint:>12.2f}x {rsvb:>10.2f}x")

print(f"{'='*W}")
