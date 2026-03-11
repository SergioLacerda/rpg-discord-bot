import json
import hashlib
from pathlib import Path

import src.infrastructure.embedding_client as client


CACHE_FILE = Path("campaign/memory/embedding_cache.json")


def _load():

    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())

    return {}


def _save(data):

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    CACHE_FILE.write_text(json.dumps(data))


def _hash(text):

    return hashlib.sha256(text.encode()).hexdigest()


def embed(text):

    cache = _load()

    key = _hash(text)

    if key in cache:
        return cache[key]

    vector = client.embed(text)

    cache[key] = vector

    _save(cache)

    return vector