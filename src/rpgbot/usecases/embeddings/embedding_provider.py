from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        pass

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:

        result = []

        for t in texts:
            vec = await self.embed(t)
            result.append(vec)

        return result