from pathlib import Path
import json
import asyncio
import hashlib
from typing import Callable, Coroutine, Any
from functools import wraps

from rpgbot.core.config import settings


def persistent_cache(
    cache_path: Path,
    key_func: Callable[[str], str] = lambda x: hashlib.sha256(x.encode()).hexdigest()
):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}

    def load():
        nonlocal cache
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except:
                cache = {}
        return cache

    async def save():
        await asyncio.to_thread(
            cache_path.write_text,
            json.dumps(cache, ensure_ascii=False, indent=None),
            encoding="utf-8"
        )

    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        @wraps(func)
        async def wrapper(text: str, *args, **kwargs):
            load() 
            key = key_func(text)
            if key in cache:
                return cache[key]

            result = await func(text, *args, **kwargs)
            cache[key] = result
            await save()
            return result

        return wrapper

    return decorator

def prune_cache(cache: dict):

    if len(cache) <= MAX_CACHE_SIZE:
        return cache

    # remove entradas mais antigas
    keys = list(cache.keys())[: len(cache) - settings.MAX_CACHE_SIZE]

    for k in keys:
        cache.pop(k, None)

    return cache