try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from rpgbot.usecases.embeddings.embedding_provider import EmbeddingProvider
from rpgbot.core.providers import embedding_registry


if AsyncOpenAI:

    class LMStudioEmbeddingProvider(EmbeddingProvider):

        def __init__(self, model: str, base_url: str):

            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key="lmstudio"
            )

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


    @embedding_registry.register("lmstudio")
    def create_lmstudio_embedding(**kwargs):

        return LMStudioEmbeddingProvider(
            model=kwargs.get("model"),
            base_url=kwargs.get("base_url")
        )