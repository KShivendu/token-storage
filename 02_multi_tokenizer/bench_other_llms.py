import os, math, time
import numpy as np
import constriction
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer

CONFIGS = {
    'qwen2.5': 'Qwen/Qwen2.5-7B',
    'deepseek-v2': 'deepseek-ai/DeepSeek-V2-Lite',
    'gemma-2': 'google/gemma-2-9b',
}

def median_p99(t):
    t = sorted(t)
    return t[len(t)//2], t[int(len(t)*0.99)]

def pack24(ids):
    arr = np.array(ids, dtype=np.uint32)
    out = np.zeros((len(arr), 3), dtype=np.uint8)
    out[:,0] = (arr >> 16) & 0xFF
    out[:,1] = (arr >> 8) & 0xFF
    out[:,2] = arr & 0xFF
    return out.tobytes()

print("Loading WikiText-103...")
ds_train = load_dataset("wikitext", "wikitext-103-v1", split="train", trust_remote_code=True)
ds_test = load_dataset("wikitext", "wikitext-103-v1", split="test", trust_remote_code=True)
articles_all = [t.strip() for t in ds_test["text"] if len(t.strip().split()) >= 30][:1773]
articles_lat = [t.strip() for t in ds_test["text"] if 50 <= len(t.strip().split()) <= 500][:200]
train_text = "\n".join(ds_train["text"][:80000])  # subset for tractable training time

for name, repo in CONFIGS.items():
    print(f"\n=== {name} ({repo}) ===")
    tok = AutoTokenizer.from_pretrained(repo)
    VOCAB = len(tok)
    bits = math.ceil(math.log2(VOCAB))
    print(f"vocab={VOCAB:,} bits_needed={bits}")

    cache = f"static_probs_{name}.npy"
    if os.path.exists(cache):
        probs = np.load(cache)
    else:
        print("training static freq table...")
        ids = tok.encode(train_text)
        counts = np.ones(VOCAB, dtype=np.int64)
        for tid, cnt in Counter(ids).items():
            counts[tid] += cnt
        probs = counts.astype(np.float64) / counts.sum()
        np.save(cache, probs)
    model = constriction.stream.model.Categorical(probs, perfect=False)

    total_raw = total_32 = total_24 = total_ans = 0
    for text in articles_all:
        raw = text.encode("utf-8")
        ids = tok.encode(text)
        total_raw += len(raw)
        total_32 += len(ids) * 4
        total_24 += len(pack24(ids))
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(np.array(ids, dtype=np.int32), model)
        total_ans += len(coder.get_compressed().tobytes())

    print(f"  uint32 raw: {total_raw/total_32:.3f}x   3-byte raw: {total_raw/total_24:.3f}x   + static ANS: {total_raw/total_ans:.3f}x")

    N_RUNS = 5
    tok_enc, tok_dec, ans_enc, ans_dec = [], [], [], []
    for text in articles_lat:
        ids = tok.encode(text)
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(np.array(ids, dtype=np.int32), model)
        compressed = coder.get_compressed().tobytes()
        n = len(ids)
        for _ in range(N_RUNS):
            t0 = time.perf_counter(); tok.encode(text); tok_enc.append((time.perf_counter()-t0)*1e6)
            t0 = time.perf_counter(); tok.decode(ids); tok_dec.append((time.perf_counter()-t0)*1e6)
            t0 = time.perf_counter()
            ids2 = tok.encode(text)
            c2 = constriction.stream.stack.AnsCoder()
            c2.encode_reverse(np.array(ids2, dtype=np.int32), model)
            c2.get_compressed().tobytes()
            ans_enc.append((time.perf_counter()-t0)*1e6)
            t0 = time.perf_counter()
            buf = np.frombuffer(compressed, dtype=np.uint32).copy()
            dec_ids = constriction.stream.stack.AnsCoder(buf).decode(model, n).tolist()
            tok.decode(dec_ids)
            ans_dec.append((time.perf_counter()-t0)*1e6)

    tem, tep = median_p99(tok_enc); tdm, tdp = median_p99(tok_dec)
    aem, aep = median_p99(ans_enc); adm, adp = median_p99(ans_dec)
    print(f"  tokenizer only: enc {tem:.1f}/{tep:.1f}us p99   dec {tdm:.1f}/{tdp:.1f}us p99")
    print(f"  + ANS:          enc {aem:.1f}/{aep:.1f}us p99   dec {adm:.1f}/{adp:.1f}us p99")
