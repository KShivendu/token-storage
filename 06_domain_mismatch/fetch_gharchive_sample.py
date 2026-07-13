"""
Fetch one real hourly GitHub Archive dump (public, no auth needed) and cache
it as gharchive_sample.json.gz in this directory, where bench_percentiles.py
expects to find it.

GH Archive (https://www.gharchive.org/) publishes every GitHub public event
as one gzipped JSON-lines file per hour. Any hour works for the domain
mismatch experiment; this pins one specific hour so results are reproducible.
"""
import os
import urllib.request

URL = "https://data.gharchive.org/2024-01-01-0.json.gz"
OUT = os.path.join(os.path.dirname(__file__), "gharchive_sample.json.gz")

if os.path.exists(OUT):
    print(f"Already cached at {OUT}")
else:
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, OUT)
    print(f"Saved to {OUT} ({os.path.getsize(OUT):,} bytes)")
