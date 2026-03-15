import httpx

from rpgbot.usecases.embeddings.embedding_provider import EmbeddingProvider
from rpgbot.core.providers import embedding_registry


class OllamaEmbeddingProvider(EmbeddingProvider):

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):

        self.model = model
        self.base_url = base_url

    async def embed(self, text):

        async with httpx.AsyncClient() as client:

            resp = await client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )

        return resp.json()["embedding"]

    async def embed_batch(self, texts):

        result = []

        for t in texts:
            vec = await self.embed(t)
            result.append(vec)

        return result


@embedding_registry.register("ollama")
def create_ollama_embedding(**kwargs):

    return OllamaEmbeddingProvider(
        model=kwargs.get("model"),
        base_url=kwargs.get("base_url", "http://localhost:11434")
    )