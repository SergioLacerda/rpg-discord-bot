import random
import hashlib
from typing import List

from openai import AsyncOpenAI

from rpgbot.config import OPENAI_API_KEY

MODEL = "text-embedding-3-small"
DIMENSION = 1536  # fixo para text-embedding-3-small

_client: AsyncOpenAI | None = None


async def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY não configurada no ambiente")
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


def deterministic_vector(text: str, dim: int = DIMENSION) -> List[float]:
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    return [rng.random() for _ in range(dim)]


async def remote_embed(text: str) -> List[float]:
    if not text or not text.strip():
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
        print(f"[remote_embed] Falha ao gerar embedding: {e}")
        return deterministic_vector(text)