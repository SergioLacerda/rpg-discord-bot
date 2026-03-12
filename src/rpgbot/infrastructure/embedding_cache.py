import json
import hashlib
from pathlib import Path
from rpgbot.infrastructure.embedding_client import embed as remote_embed

CACHE_PATH = Path("campaign/memory/embedding_cache.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

_cache = None


def load_cache():
    global _cache
    if _cache is None:
        if CACHE_PATH.exists():
            _cache = json.loads(CACHE_PATH.read_text())
        else:
            _cache = {}
    return _cache


def save_cache():
    CACHE_PATH.write_text(json.dumps(_cache))


def embed(text: str):

    cache = load_cache()

    key = hashlib.sha256(text.encode()).hexdigest()

    if key in cache:
        return cache[key]

    vector = remote_embed(text)

    cache[key] = vector
    save_cache()

    return vector