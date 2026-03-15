try:
    import google.generativeai as genai
except ImportError:
    genai = None

from rpgbot.usecases.embeddings.embedding_provider import EmbeddingProvider
from rpgbot.core.providers import embedding_registry


if genai:

    class GeminiEmbeddingProvider(EmbeddingProvider):

        def __init__(self, api_key: str, model: str):

            genai.configure(api_key=api_key)
            self.model = model

        async def embed(self, text):

            resp = genai.embed_content(
                model=self.model,
                content=text
            )

            return resp["embedding"]

        async def embed_batch(self, texts):

            vectors = []

            for t in texts:
                vectors.append(await self.embed(t))

            return vectors


    @embedding_registry.register("gemini")
    def create_gemini_embedding(**kwargs):

        return GeminiEmbeddingProvider(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model")
        )