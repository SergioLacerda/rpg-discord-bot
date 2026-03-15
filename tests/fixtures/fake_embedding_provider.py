
class FakeEmbeddingProvider:

    async def embed(self, text):
        return [0.1] * 32
