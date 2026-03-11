from openai import OpenAI
from src.config import OPENAI_API_KEY


MODEL = "text-embedding-3-small"


_client = None


def get_client():

    global _client

    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)

    return _client


def embed(text: str):

    client = get_client()

    return client.embeddings.create(
        model=MODEL,
        input=text
    ).data[0].embedding