import random
from typing import List

from openai import AsyncOpenAI

from rpgbot.core.config import settings
from rpgbot.core.resilience import resilient_call
from rpgbot.utils.hash_utils import sha256_hash


MODEL = "text-embedding-3-small"
DIMENSION = 1536

_client: AsyncOpenAI | None = None


async def get_client() -> AsyncOpenAI:

    global _client

    if _client is None:

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY não configurada")

        _client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            max_retries=0,
            timeout=20,
        )

    return _client


def deterministic_vector(text: str, dim: int = DIMENSION) -> List[float]:
    """Fallback determinístico quando a API falha"""

    seed = int(sha256_hash(text.encode()), 16) % (2**32)

    rng = random.Random(seed)

    return [rng.random() for _ in range(dim)]


async def _openai_embed(text: str) -> List[float]:

    client = await get_client()

    resp = await client.embeddings.create(
        model=MODEL,
        input=text,
        encoding_format="float",
    )

    return resp.data[0].embedding


async def remote_embed(text: str) -> List[float]:
    if not text.strip():
        return deterministic_vector(text)

    return await resilient_call(
        [
            _openai_embed,
            deterministic_vector
        ],
        text,
        speculative_delay=0.2
    )