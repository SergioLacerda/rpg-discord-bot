import hashlib
import logging
from typing import Optional, Callable, Awaitable

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, retry_if_exception_type

from rpgbot.infrastructure.embedding_client import get_client
from rpgbot.usecases.retrieve_context import hierarchical_context
from rpgbot.infrastructure.response_cache import get_cached_response, set_cached_response
from rpgbot.utils.concurrency.deduplicate_async import InflightDeduplicator

llm_deduplicator = InflightDeduplicator()

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
MAX_TOKENS = 300
TEMPERATURE = 0.8
TIMEOUT = 20
MAX_RETRIES = 2


def _should_retry(exc: Exception) -> bool:

    if isinstance(exc, RateLimitError):

        if hasattr(exc, "body") and exc.body:

            code = exc.body.get("error", {}).get("code")

            if code == "insufficient_quota":
                return False

        return True

    if isinstance(exc, APITimeoutError):
        return True

    if isinstance(exc, APIError):
        return True

    return False


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
    retry=retry_if_exception(_should_retry),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        "Retry %s/%s após erro: %s",
        retry_state.attempt_number,
        MAX_RETRIES,
        retry_state.outcome.exception())
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

    key = hashlib.sha256(prompt.encode()).hexdigest()

    async def _generate():

        if cached := await cache_get(prompt):
            return cached

        local_client = client

        if local_client is None:
            local_client = await get_client()

        try:

            content = await _call_openai(prompt, local_client)

            await cache_set(prompt, content)

            return content

        except RateLimitError as e:

            if hasattr(e, "body") and e.body:

                code = e.body.get("error", {}).get("code")

                if code == "insufficient_quota":

                    logger.error("Quota esgotada")

                    return "⚠️ O mestre está sem mana."

            raise

        except Exception:

            logger.exception("Falha após retries")

            return "⚠️ O ambiente parece congelado por um momento..."

    return await llm_deduplicator.run(key, _generate)
    