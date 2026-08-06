import hashlib
import json
import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)   # make the cache folder if it doesn't exist

def _cache_key(text, n, temperature):
    """A fingerprint of the inputs that determine the result."""
    raw = f"{text}|n={n}|t={temperature}"
    return hashlib.sha256(raw.encode()).hexdigest()

def get_cached(text, n, temperature):
    """Return the stored result if we've seen this exact input before, else None."""
    key = _cache_key(text, n, temperature)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_cache(text, n, temperature, result):
    """Store a result under its input fingerprint."""
    key = _cache_key(text, n, temperature)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)