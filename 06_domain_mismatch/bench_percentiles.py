"""
Compression ratio percentiles — two corpora:

  1. WikiText-103 test articles  (natural language text)
  2. GitHub Archive events       (real-world JSON API payloads)

For JSON, tests whether a JSON-corpus-trained ANS table beats gzip,
unlike the text-trained table which has wrong priors for JSON structure.

Methods: lz4, gzip-9, zstd-22, zstd-dict, brotli-11, raw uint16/32,
         r50k+static ANS (text-trained), r50k+static ANS (JSON-trained),
         msgpack+gzip
"""

import argparse, gzip, os, subprocess, json
import lz4.frame
import msgpack
import zstandard as zstd
import brotli
import tiktoken
import numpy as np
import constriction
from collections import Counter
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", choices=["text", "json"], nargs="+", default=["text", "json"],
    help="Run only these datasets/phases (default: both). 'text' = WikiText percentiles/histogram, "
         "'json' = GitHub Archive JSON benchmark.",
)
parser.add_argument(
    "--only", nargs="+", default=None,
    help="Only compute/report these method keys (e.g. tok_zstd_dict raw_u16). "
         "Default: all methods. See METHODS/JSON_METHODS lists for valid keys.",
)
args = parser.parse_args()
PHASES = set(args.dataset)
ONLY = set(args.only) if args.only else None

def want(key):
    """True if this method key should be computed/reported, given --only."""
    return ONLY is None or key in ONLY

def save_json_merged(path, new_data):
    """Write new_data to path, merging with any existing content instead of
    clobbering it. Matters when --only runs a subset: we don't want a partial
    run to wipe out method results from a previous full run."""
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.update(new_data)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

CACHE_DIR = os.path.dirname(__file__)
VOCAB_R50K   = 50_257
VOCAB_CL100K = 100_277
TOKENIZERS = {"r50k": ("r50k_base", VOCAB_R50K), "cl100k": ("cl100k_base", VOCAB_CL100K)}

# ── static freq tables (train split, cached) ──────────────────────────────────

def load_static_probs(name, enc_name, vocab_size):
    cache = os.path.join(CACHE_DIR, f"static_probs_{name}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    print(f"  {name}: tokenizing WikiText-103 train split...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    enc = tiktoken.get_encoding(enc_name)
    ids = enc.encode("\n".join(ds["text"]))
    counts = np.ones(vocab_size, dtype=np.int64)
    for tok_id, cnt in Counter(ids).items():
        counts[tok_id] += cnt
    probs = counts.astype(np.float64) / counts.sum()
    np.save(cache, probs)
    return probs

print("Loading static freq tables...")
static_probs, encs, static_models = {}, {}, {}
for name, (enc_name, vocab_size) in TOKENIZERS.items():
    static_probs[name] = load_static_probs(name, enc_name, vocab_size)
    encs[name] = tiktoken.get_encoding(enc_name)
    static_models[name] = constriction.stream.model.Categorical(static_probs[name], perfect=False)

# ── load zstd dict ─────────────────────────────────────────────────────────────

has_zstd_dict = False
if want("zstd_dict"):
    dict_path = os.path.join(CACHE_DIR, "zstd_dict_112k.bin")
    if os.path.exists(dict_path):
        with open(dict_path, "rb") as f:
            zstd_dict_data = f.read()
        zdict = zstd.ZstdCompressionDict(zstd_dict_data)
        zstd_dict_c = zstd.ZstdCompressor(level=22, dict_data=zdict)
        has_zstd_dict = True
    else:
        print("  Warning: zstd dict not found, skipping")

zstd_c = zstd.ZstdCompressor(level=22)

# ── zstd dict trained on packed token-ID bytes, not raw text ───────────────────
# Same idea as zstd_dict_112k.bin, but the training samples are r50k token IDs
# packed as big-endian uint16, so the dictionary can capture repeated token
# *phrases* (multi-token n-grams) instead of repeated byte/character patterns.

import struct

if want("tok_zstd_dict"):
    tok_dict_path = os.path.join(CACHE_DIR, "zstd_dict_token_112k.bin")
    if os.path.exists(tok_dict_path):
        with open(tok_dict_path, "rb") as f:
            tok_dict_data = f.read()
    else:
        print("Training zstd dictionary on packed token-ID bytes (WikiText-103 train)...")
        ds_train = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
        train_texts = [t.strip() for t in ds_train["text"] if len(t.strip().split()) >= 30]
        samples = []
        for t in train_texts:
            ids = encs["r50k"].encode(t)
            if ids:
                samples.append(struct.pack(f">{len(ids)}H", *ids))
        d = zstd.train_dictionary(112 * 1024, samples)
        tok_dict_data = d.as_bytes()
        with open(tok_dict_path, "wb") as f:
            f.write(tok_dict_data)
        print(f"  Trained on {len(samples)} samples, saved to {tok_dict_path}")

    tok_zdict = zstd.ZstdCompressionDict(tok_dict_data)
    zstd_tokdict_c = zstd.ZstdCompressor(level=22, dict_data=tok_zdict)

# ── zlib preset dictionary: naive, not trained (RFC 1950 zdict, capped at 32KB) ─
# No ZDICT-style construction here, just take up to 32KB of raw train-split
# samples verbatim as a "preset dictionary" primer for DEFLATE's LZ77 window.
# This isolates whether zstd's trained-dictionary *algorithm* is what mattered,
# or whether any shared context helps about as much.

import zlib

ZLIB_DICT_SIZE = 32 * 1024  # zlib's window cap

def build_zlib_preset_dict(samples, cap=ZLIB_DICT_SIZE):
    buf = bytearray()
    for s in samples:
        if len(buf) >= cap:
            break
        buf += s
    return bytes(buf[:cap])

if want("zlib_presetdict") or want("tok_zlib_presetdict"):
    ds_train_zlib = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    train_texts_zlib = [t.strip() for t in ds_train_zlib["text"] if len(t.strip().split()) >= 30]

    if want("zlib_presetdict"):
        zlib_text_dict = build_zlib_preset_dict([t.encode("utf-8") for t in train_texts_zlib])

    if want("tok_zlib_presetdict"):
        zlib_tok_dict = build_zlib_preset_dict([
            struct.pack(f">{len(ids)}H", *ids)
            for ids in (encs["r50k"].encode(t) for t in train_texts_zlib)
            if ids
        ])

def zlib_compress(data, zdict):
    co = zlib.compressobj(level=9, wbits=15, zdict=zdict)
    return co.compress(data) + co.flush()

# ── compress one doc ───────────────────────────────────────────────────────────

def ans_encode_size(ids, model):
    enc = constriction.stream.stack.AnsCoder()
    enc.encode_reverse(np.array(ids, dtype=np.int32), model)
    return len(enc.get_compressed().tobytes())

def compress_doc(text):
    raw = text.encode("utf-8")
    n = len(raw)
    if n == 0:
        return None

    r50k_ids = encs["r50k"].encode(text)
    if not r50k_ids:
        return None

    result = {"bytes": n, "words": len(text.split())}

    if want("lz4"):
        result["lz4"] = n / len(lz4.frame.compress(raw, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX))
    if want("gzip"):
        result["gzip"] = n / len(gzip.compress(raw, compresslevel=9))
    if want("zstd"):
        result["zstd"] = n / len(zstd_c.compress(raw))
    if want("brotli"):
        result["brotli"] = n / len(brotli.compress(raw, quality=11))
    if want("raw_u16"):
        result["raw_u16"] = n / (len(r50k_ids) * 2)  # r50k fits in uint16
    if want("raw_u32") or want("raw_u24"):
        cl100k_ids = encs["cl100k"].encode(text)
        if want("raw_u32"):
            result["raw_u32"] = n / (len(cl100k_ids) * 4)  # cl100k needs uint32
        if want("raw_u24"):
            result["raw_u24"] = n / (len(cl100k_ids) * 3)  # cl100k also fits in 3 bytes
    if want("ans"):
        result["ans"] = n / ans_encode_size(r50k_ids, static_models["r50k"])
    if want("tok_zstd_dict") or want("tok_lz4") or want("tok_gzip") or want("tok_zlib_presetdict"):
        r50k_packed = struct.pack(f">{len(r50k_ids)}H", *r50k_ids)
        if want("tok_zstd_dict"):
            result["tok_zstd_dict"] = n / len(zstd_tokdict_c.compress(r50k_packed))
        if want("tok_lz4"):
            result["tok_lz4"] = n / len(lz4.frame.compress(r50k_packed, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX))
        if want("tok_gzip"):
            result["tok_gzip"] = n / len(gzip.compress(r50k_packed, compresslevel=9))
        if want("tok_zlib_presetdict"):
            result["tok_zlib_presetdict"] = n / len(zlib_compress(r50k_packed, zlib_tok_dict))
    if want("zlib_presetdict"):
        result["zlib_presetdict"] = n / len(zlib_compress(raw, zlib_text_dict))
    if has_zstd_dict:
        result["zstd_dict"] = n / len(zstd_dict_c.compress(raw))
    return result

if "text" in PHASES:
    # ── run over test split ────────────────────────────────────────────────────────

    print("Loading WikiText-103 test split...")
    ds_test = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test")
    articles = [t.strip() for t in ds_test["text"] if len(t.strip().split()) >= 30]
    print(f"  Articles (>=30 words): {len(articles)}")

    results = []
    for i, text in enumerate(articles):
        r = compress_doc(text)
        if r:
            results.append(r)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(articles)}...")

    print(f"  Done. {len(results)} articles compressed.")

    # ── results table ──────────────────────────────────────────────────────────────

    METHODS = [
        ("lz4",       "lz4 max"),
        ("gzip",      "gzip-9"),
        ("zstd",      "zstd-22"),
        ("brotli",    "brotli-11"),
        ("raw_u32",   "raw uint32 (cl100k)"),
        ("raw_u24",   "raw uint24 (cl100k, 3B)"),
        ("raw_u16",   "raw uint16 (r50k)"),
        ("tok_lz4",   "r50k uint16 + lz4 (no dict)"),
        ("tok_gzip",  "r50k uint16 + gzip (no dict)"),
        ("zlib_presetdict",     "zlib + naive preset dict 32KB (text)"),
        ("tok_zlib_presetdict", "r50k uint16 + naive preset dict 32KB"),
        ("tok_zstd_dict", "r50k uint16 + zstd token-dict 112KB"),
        ("ans", "r50k + static ANS"),
    ]
    if has_zstd_dict:
        METHODS.insert(3, ("zstd_dict", "zstd-dict 112KB"))
    METHODS = [(k, l) for k, l in METHODS if want(k)]

    print(f"\n{'='*64}")
    print(f"  Compression ratio — WikiText-103 test ({len(results)} articles)")
    print(f"{'='*64}")
    print(f"  {'Method':<26}  {'min':>6}  {'median':>6}  {'max':>6}")
    print(f"  {'-'*56}")

    summary = {}
    for key, label in METHODS:
        vals = [r[key] for r in results if key in r]
        mn  = min(vals)
        med = float(np.median(vals))
        mx  = max(vals)
        summary[key] = {"label": label, "min": mn, "median": med, "max": mx}
        print(f"  {label:<26}  {mn:>5.2f}x  {med:>6.2f}x  {mx:>5.2f}x")

    print(f"{'='*64}")

    save_json_merged(os.path.join(CACHE_DIR, "bench_percentiles_results.json"), summary)
    print("\nResults saved to bench_percentiles_results.json")

    # ── block-level LZ4 (ES-style 16KB blocks, multiple docs share a window) ────────
    # The per-doc "lz4" ratio above understates real-world LZ4: Elasticsearch and
    # similar stores pack ~16KB of documents into one block and compress the whole
    # block together, so LZ4 can back-reference across docs, not just within one.

    if want("lz4_block"):
        BLOCK_SIZE = 16 * 1024
        blocks, current_block, current_size = [], [], 0
        for text in articles:
            raw = text.encode("utf-8")
            if current_size + len(raw) > BLOCK_SIZE and current_block:
                blocks.append(current_block)
                current_block, current_size = [], 0
            current_block.append(raw)
            current_size += len(raw)
        if current_block:
            blocks.append(current_block)

        block_ratios = []
        for block in blocks:
            concat = b"".join(block)
            comp = lz4.frame.compress(concat, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)
            block_ratios.append(len(concat) / len(comp))

        block_summary = {
            "label": "lz4 max, 16KB blocks (ES-style)",
            "min": float(min(block_ratios)),
            "median": float(np.median(block_ratios)),
            "max": float(max(block_ratios)),
            "n_blocks": len(blocks),
            "docs_per_block_avg": len(articles) / len(blocks),
        }
        print(f"\n  LZ4 block-level ({len(blocks)} blocks, ~{block_summary['docs_per_block_avg']:.1f} docs/block):")
        print(f"  {'lz4 max, 16KB blocks':<26}  {block_summary['min']:>5.2f}x  {block_summary['median']:>6.2f}x  {block_summary['max']:>5.2f}x")

        summary["lz4_block"] = block_summary
        save_json_merged(os.path.join(CACHE_DIR, "bench_percentiles_results.json"), summary)

    # ── histogram bins for blog chart ─────────────────────────────────────────────
    # Bins: [0.95, 1.05, 1.15, ..., 4.45] (centres spaced 0.1 apart)
    HIST_METHODS = ["gzip", "zstd_dict", "raw_u16", "ans"]
    HIST_LABELS  = ["gzip -9", "zstd dict 112KB", "r50k uint16 raw", "r50k + static ANS"]
    bin_edges = np.arange(0.9, 4.55, 0.1)
    bin_centres = ((bin_edges[:-1] + bin_edges[1:]) / 2).round(2).tolist()

    hist_out = {"bin_centres": bin_centres}
    for key, label in zip(HIST_METHODS, HIST_LABELS):
        if not want(key):
            continue
        vals = np.array([r[key] for r in results if key in r])
        if len(vals) == 0:
            continue
        counts, _ = np.histogram(vals, bins=bin_edges)
        hist_out[key] = {"label": label, "counts": counts.tolist()}

    save_json_merged(os.path.join(CACHE_DIR, "bench_hist_results.json"), hist_out)
    print("Histogram data saved to bench_hist_results.json")

if "json" in PHASES:
    # ══════════════════════════════════════════════════════════════════════════════
    # PHASE 2 — JSON API payloads (GitHub Archive)
    # Hypothesis: ANS with a JSON-trained table beats gzip on JSON,
    #             because structural tokens become high-frequency and get short codes.
    # ══════════════════════════════════════════════════════════════════════════════

    GH_URL   = "https://data.gharchive.org/2024-01-15-14.json.gz"
    GH_CACHE = os.path.join(CACHE_DIR, "gharchive_sample.json.gz")
    MAX_JSON_EVENTS = 10_000

    print(f"\n{'='*64}")
    print("PHASE 2 — GitHub Archive JSON events")
    print(f"{'='*64}")

    if not os.path.exists(GH_CACHE):
        print("Downloading GitHub Archive sample (~138 MB)...")
        subprocess.run(["curl", "-L", "-o", GH_CACHE, GH_URL], check=True)

    print(f"Parsing events (up to {MAX_JSON_EVENTS})...")
    json_events = []
    with gzip.open(GH_CACHE, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                json_events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
            if len(json_events) >= MAX_JSON_EVENTS:
                break

    json_docs = [json.dumps(e, separators=(",", ":")).encode("utf-8") for e in json_events]
    print(f"  {len(json_docs)} events, median {int(np.median([len(d) for d in json_docs]))} bytes")

    # ── train JSON-specific ANS table ─────────────────────────────────────────────

    json_probs_path = os.path.join(CACHE_DIR, "static_probs_r50k_json.npy")
    if os.path.exists(json_probs_path):
        print("Loading JSON-trained ANS table (cached)...")
        json_probs = np.load(json_probs_path)
    else:
        print("Training JSON-specific ANS table on corpus...")
        counts = np.ones(VOCAB_R50K, dtype=np.int64)
        for doc in json_docs:
            for tok_id, cnt in Counter(encs["r50k"].encode(doc.decode("utf-8", errors="replace"))).items():
                counts[tok_id] += cnt
        json_probs = counts.astype(np.float64) / counts.sum()
        np.save(json_probs_path, json_probs)

    json_model = constriction.stream.model.Categorical(json_probs, perfect=False)

    def ans_size_with_header(ids, model):
        """ANS compressed size + 4-byte token-count header. Table is shared (no per-doc overhead)."""
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(np.array(ids, dtype=np.int32), model)
        return 4 + len(coder.get_compressed().tobytes())

    # ── benchmark ─────────────────────────────────────────────────────────────────

    print("Benchmarking JSON methods...")
    json_results = []
    for i, (raw, event) in enumerate(zip(json_docs, json_events)):
        n = len(raw)
        if n == 0:
            continue
        ids = encs["r50k"].encode(raw.decode("utf-8", errors="replace"))
        if not ids:
            continue
        mp = msgpack.packb(event, use_bin_type=True)
        r = {
            "n": n,
            "lz4":              n / len(lz4.frame.compress(raw, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)),
            "gzip":             n / len(gzip.compress(raw, compresslevel=9)),
            "msgpack_gzip":     n / len(gzip.compress(mp, compresslevel=9)),
            "tok_raw_u16":      n / (len(ids) * 2),
            "ans_text_trained": n / ans_size_with_header(ids, static_models["r50k"]),
            "ans_json_trained": n / ans_size_with_header(ids, json_model),
        }
        json_results.append(r)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(json_docs)}...")

    print(f"  Done. {len(json_results)} docs.")

    JSON_METHODS = [
        ("lz4",              "JSON + LZ4 max"),
        ("gzip",             "JSON + gzip -9"),
        ("msgpack_gzip",     "MessagePack + gzip -9"),
        ("tok_raw_u16",      "JSON tok uint16 (raw)"),
        ("ans_text_trained", "JSON tok+ANS (text-trained)"),
        ("ans_json_trained", "JSON tok+ANS (JSON-trained)"),
    ]
    JSON_METHODS = [(k, l) for k, l in JSON_METHODS if want(k)]

    print(f"\n{'='*64}")
    print(f"  JSON compression — {len(json_results)} GitHub Archive events")
    print(f"  (ratio = raw JSON bytes / compressed bytes, shared static table)")
    print(f"{'='*64}")
    print(f"  {'Method':<34}  {'p5':>5}  {'median':>6}  {'p95':>5}")
    print(f"  {'-'*58}")

    json_summary = {}
    for key, label in JSON_METHODS:
        vals = [r[key] for r in json_results if key in r]
        p5  = float(np.percentile(vals, 5))
        med = float(np.median(vals))
        p95 = float(np.percentile(vals, 95))
        json_summary[key] = {"label": label, "p5": p5, "median": med, "p95": p95}
        print(f"  {label:<34}  {p5:>4.2f}x  {med:>5.2f}x  {p95:>5.2f}x")

    print(f"{'='*64}")

    save_json_merged(os.path.join(CACHE_DIR, "bench_json_results.json"), json_summary)
    print("JSON results saved to bench_json_results.json")
