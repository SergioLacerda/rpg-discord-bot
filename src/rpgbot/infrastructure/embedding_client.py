import random
import hashlib

from openai import OpenAI
from rpgbot.config import OPENAI_API_KEY

MODEL = "text-embedding-3-small"
_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def deterministic_vector(text, dim=1536):

    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    rng = random.Random(seed)

    return [rng.random() for _ in range(dim)]


def embed(text: str):

    try:
        client = get_client()
        return client.embeddings.create(
            model=MODEL,
            input=text
        ).data[0].embedding

    except Exception:
        # fallback simples
        return deterministic_vector(text)