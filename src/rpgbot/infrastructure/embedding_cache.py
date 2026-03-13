import random
import asyncio
import json
import logging
from pathlib import Path

from rpgbot.infrastructure.embedding_client import remote_embed, deterministic_vector
from rpgbot.utils import persistent_cache, embedding_key
from rpgbot.utils.concurrency.deduplicate_async import InflightDeduplicator


logger = logging.getLogger(__name__)

CACHE_PATH = Path("campaign/memory/embedding_cache.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_ID = "remote-v1"

_cache: dict | None = None

deduplicator = InflightDeduplicator()


def _load_cache_sync() -> dict:

    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Erro ao carregar cache: {e}")

    return {}


def _save_cache_sync(cache_data: dict) -> None:

    try:
        CACHE_PATH.write_text(
            json.dumps(cache_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"Erro ao salvar cache: {e}")


async def load_cache() -> dict:

    global _cache

    if _cache is None:
        _cache = await asyncio.to_thread(_load_cache_sync)

    return _cache


async def save_cache() -> None:

    if _cache is not None:
        await asyncio.to_thread(_save_cache_sync, _cache)


@persistent_cache(CACHE_PATH)
async def embed(text: str) -> list[float]:

    if not text.strip():
        return deterministic_vector(text)

    normalized = " ".join(text.lower().split())

    key = embedding_key(f"{MODEL_ID}:{normalized}")

    async def _generate():

        try:
            return await asyncio.wait_for(remote_embed(text), timeout=20)

        except asyncio.CancelledError:
            raise

        except Exception as e:
            logger.warning(f"[embed] fallback determinístico ativado: {e}")
            return deterministic_vector(text)

    return await deduplicator.run(key, _generate)