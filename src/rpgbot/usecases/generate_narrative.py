import hashlib
import logging
from typing import Optional, Callable, Awaitable

from openai import AsyncOpenAI, RateLimitError

from rpgbot.infrastructure.narrative_memory import memory
from rpgbot.infrastructure.embedding_client import get_client
from rpgbot.usecases.retrieve_context import hierarchical_context
from rpgbot.infrastructure.response_cache import get_cached_response, set_cached_response
from rpgbot.utils.concurrency.deduplicate_async import InflightDeduplicator
from rpgbot.utils.text.normalize_utils import compress_context
from rpgbot.core.resilience import resilient_call

llm_deduplicator = InflightDeduplicator()

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
MAX_TOKENS = 300
TEMPERATURE = 0.8
TIMEOUT = 20


async def _openai_chat(prompt: str, client: AsyncOpenAI) -> str:

    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Mestre de RPG narrativo imparcial."},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT,
    )

    content = resp.choices[0].message.content.strip()

    if not content:
        raise ValueError("Resposta vazia")

    return content


def _fallback_narrative(prompt: str) -> str:

    logger.warning("LLM indisponível → fallback narrativo")

    return "⚠️ O mundo parece hesitar por um instante... algo está prestes a acontecer."


async def build_prompt(action: str, ctx_provider):

    ctx = await ctx_provider(action)

    ctx = compress_context(ctx)

    history = memory.get()

    return f"""
Você é um mestre de RPG narrativo experiente e imparcial.

Resumo da história até agora:
{history}

Contexto relevante:
{ctx}

Ação do jogador:
{action}

Descreva o que acontece em seguida.
"""

async def generate_narrative(
    action: str,
    *,
    client: Optional[AsyncOpenAI] = None,
    ctx_provider: Callable[[str], Awaitable[str]] = hierarchical_context,
    cache_get: Callable[[str], Awaitable[Optional[str]]] = get_cached_response,
    cache_set: Callable[[str, str], Awaitable[None]] = set_cached_response,
) -> str:

    prompt = await build_prompt(action, ctx_provider)

    key = hashlib.sha256(prompt.encode()).hexdigest()

    async def _generate():

        if cached := await cache_get(prompt):
            return cached

        local_client = client or await get_client()

        try:

            content = await resilient_call(
                [
                    lambda p: _openai_chat(p, local_client),
                    _fallback_narrative
                ],
                prompt,
                speculative_delay=0.4
            )

            await cache_set(prompt, content)

            if not content.startswith("⚠️"):
                memory.update(content)

            return content

        except RateLimitError as e:

            if hasattr(e, "body") and e.body:
                code = e.body.get("error", {}).get("code")

                if code == "insufficient_quota":
                    logger.error("Quota esgotada")
                    return "⚠️ O mestre está sem mana."

            raise

        except Exception:
            logger.exception("Falha total no sistema narrativo")
            return "⚠️ O ambiente parece congelado por um momento..."

    return await llm_deduplicator.run(key, _generate)