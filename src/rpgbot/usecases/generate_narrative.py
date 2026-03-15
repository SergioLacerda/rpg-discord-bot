import logging
from typing import Optional, Callable, Awaitable

from rpgbot.core.cache.narrative_cache import NarrativeLRUCache
from rpgbot.usecases.retrieve_context import hierarchical_context
from rpgbot.usecases.narrative_engine import NarrativeEngine

from rpgbot.infrastructure.narrative_memory import memory
from rpgbot.infrastructure.response_cache import (
    get_cached_response,
    set_cached_response,
)

from rpgbot.utils.text.normalize_utils import compress_context
from rpgbot.utils.hash_utils import sha256_hash


logger = logging.getLogger(__name__)

_engine = NarrativeEngine()

__all__ = ["generate_narrative", "build_prompt"]

# ---------------------------------------------------------
# in-memory narrative cache
# ---------------------------------------------------------

_narrative_cache = NarrativeLRUCache(
    max_size=512,
    ttl=600
)

# ---------------------------------------------------------
# prompt builder (compatibilidade testes antigos)
# ---------------------------------------------------------

async def build_prompt(action: str, ctx_provider, memory_context=""):

    ctx = await ctx_provider(action)

    ctx = compress_context(ctx)

    history = memory.get()

    return f"""
Você é um mestre de RPG narrativo imparcial.

História:
{history}

Memória recente:
{memory_context}

Contexto:
{ctx}

Ação:
{action}
"""

# ---------------------------------------------------------
# narrative generation
# ---------------------------------------------------------

async def generate_narrative(
    action: str,
    *,
    client: Optional[AsyncOpenAI] = None,
    ctx_provider: Callable[[str], Awaitable[str]] = hierarchical_context,
    cache_get: Callable = get_cached_response,
    cache_set: Callable = set_cached_response,
    index=None,
) -> str:

    try:

        prompt_key = sha256_hash(action)

        # ---------------------------------------
        # cache
        # ---------------------------------------

        cached = await cache_get(prompt_key)

        if cached:
            return cached

        # ---------------------------------------
        # engine
        # ---------------------------------------

        result = await _engine.generate(
            action,
            ctx_provider=ctx_provider,
            index=index,
        )

        await cache_set(prompt_key, result)

        return result

    except Exception:

        logger.exception("Falha no generate_narrative")

        return "⚠️ O ambiente parece congelado por um momento..."

