import json
import hashlib
from pathlib import Path
import asyncio
from typing import Optional

CACHE_PATH = Path("campaign/memory/response_cache.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

_cache: Optional[dict] = None


def _load_cache_sync() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ResponseCache] Erro ao carregar cache: {e}")
    return {}


def _save_cache_sync(cache_data: dict) -> None:
    try:
        CACHE_PATH.write_text(
            json.dumps(cache_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        print(f"[ResponseCache] Erro ao salvar cache: {e}")


async def load_response_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = await asyncio.to_thread(_load_cache_sync)
    return _cache


async def save_response_cache() -> None:
    if _cache is not None:
        await asyncio.to_thread(_save_cache_sync, _cache)


async def get_cached_response(prompt: str) -> Optional[str]:
    cache = await load_response_cache()
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache.get(key)


async def set_cached_response(prompt: str, response: str) -> None:
    cache = await load_response_cache()
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache[key] = response
    await save_response_cache()