import json
import hashlib
from pathlib import Path
import asyncio
from typing import List

from rpgbot.infrastructure.embedding_client import remote_embed

CACHE_PATH = Path("campaign/memory/embedding_cache.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

_cache: dict | None = None


def _load_cache_sync() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Erro ao carregar cache: {e}")
    return {}


def _save_cache_sync(cache_data: dict) -> None:
    try:
        CACHE_PATH.write_text(
            json.dumps(cache_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        print(f"Erro ao salvar cache: {e}")


async def load_cache() -> dict:
    global _cache

    if _cache is None:
        _cache = await asyncio.to_thread(_load_cache_sync)

    return _cache


async def save_cache() -> None:
    if _cache is not None:
        await asyncio.to_thread(_save_cache_sync, _cache)


async def embed(text: str) -> List[float]:
    cache = await load_cache()
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if key in cache:
        return cache[key]

    vector = await remote_embed(text)

    cache[key] = vector
    await save_cache()

    return vector