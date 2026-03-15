try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from rpgbot.usecases.embeddings.embedding_provider import EmbeddingProvider
from rpgbot.core.providers import embedding_registry


if AsyncOpenAI:

    class OpenAIEmbeddingProvider(EmbeddingProvider):

        def __init__(self, api_key: str, model: str):

            self.client = AsyncOpenAI(api_key=api_key)
            self.model = model

        async def embed(self, text):

            resp = await self.client.embeddings.create(
                model=self.model,
                input=text
            )

            return resp.data[0].embedding

        async def embed_batch(self, texts):

            resp = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            return [x.embedding for x in resp.data]


    @embedding_registry.register("openai")
    def create_openai_embedding(**kwargs):

        return OpenAIEmbeddingProvider(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "text-embedding-3-small")
        )