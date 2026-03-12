import random
import hashlib
from typing import List

from openai import AsyncOpenAI

from rpgbot.config import OPENAI_API_KEY

MODEL = "text-embedding-3-small"
DIMENSION = 1536  # fixo para text-embedding-3-small

_client: AsyncOpenAI | None = None


async def get_client() -> AsyncOpenAI:
    """Retorna cliente OpenAI (lazy init)"""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY não configurada")
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


def deterministic_vector(text: str, dim: int = DIMENSION) -> List[float]:
    """Fallback determinístico quando a API falha"""
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


async def remote_embed(text: str) -> List[float]:
    """Gera embedding com OpenAI (async) ou fallback"""
    if not text.strip():
        return deterministic_vector(text)

    try:
        client = await get_client()
        resp = await client.embeddings.create(
            model=MODEL,
            input=text,
            encoding_format="float",
        )
        return resp.data[0].embedding

    except Exception as e:
        print(f"[remote_embed] Falha: {e}")
        return deterministic_vector(text)