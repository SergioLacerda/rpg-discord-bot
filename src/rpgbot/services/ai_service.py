import logging
from typing import Optional, Callable, Awaitable

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from rpgbot.infrastructure.embedding_client import get_client
from rpgbot.services.memory_service import hierarchical_context
from rpgbot.infrastructure.response_cache import get_cached_response, set_cached_response

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
MAX_TOKENS = 300
TEMPERATURE = 0.8
TIMEOUT = 20
MAX_RETRIES = 3


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
    reraise=True
)
async def _call_openai(prompt: str, client: AsyncOpenAI) -> str:

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


async def build_prompt(action: str, ctx_provider: Callable[[str], Awaitable[str]]) -> str:

    ctx = await ctx_provider(action)

    return f"""
Você é um mestre de RPG narrativo experiente e imparcial.

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

    if cached := await cache_get(prompt):
        return cached

    if client is None:
        client = await get_client()

    try:

        content = await _call_openai(prompt, client)

        await cache_set(prompt, content)

        return content

    except Exception:
        logger.exception("Falha após retries")
        return "⚠️ O ambiente parece congelado por um momento..."