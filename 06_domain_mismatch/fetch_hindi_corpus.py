"""
Lightweight real Hindi corpus via the Wikipedia REST API directly (random
articles + plaintext extracts), no full dataset download. Caches to disk.
"""
import json
import os
import time
import urllib.request
import urllib.parse

CACHE = os.path.join(os.path.dirname(__file__), "hindi_wiki_sample.json")
N_ARTICLES = 300

def api_get(params, retries=5):
    url = "https://hi.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, headers={"User-Agent": "blog-research-script/1.0 (personal blog experiment)"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise

def fetch_batch(n):
    data = api_get({
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": n,
        "format": "json",
    })
    titles = [p["title"] for p in data["query"]["random"]]
    joined = "|".join(titles)
    data2 = api_get({
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": joined,
        "format": "json",
    })
    out = []
    for page in data2["query"]["pages"].values():
        text = page.get("extract", "")
        if text:
            out.append(text)
    return out

if os.path.exists(CACHE):
    with open(CACHE, encoding="utf-8") as fh:
        articles = json.load(fh)
    print(f"Loaded {len(articles)} cached Hindi articles")
else:
    articles = []
    while len(articles) < N_ARTICLES:
        batch = fetch_batch(20)
        articles.extend(batch)
        print(f"  fetched {len(articles)} so far...")
        time.sleep(2)
    with open(CACHE, "w", encoding="utf-8") as fh:
        json.dump(articles, fh, ensure_ascii=False)
    print(f"Fetched and cached {len(articles)} Hindi articles")

total_chars = sum(len(a) for a in articles)
print(f"Total: {total_chars:,} chars across {len(articles)} articles")
