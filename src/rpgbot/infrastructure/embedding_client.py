import inspect
import random
from typing import Iterable, List

from rpgbot.core.container import container
from rpgbot.core.resilience import resilient_call
from rpgbot.utils.hash_utils import sha256_hash


DEFAULT_DIMENSION = 1536


# ---------------------------------------------------------
# deterministic fallback
# ---------------------------------------------------------

def deterministic_vector(text: str, dim: int = DEFAULT_DIMENSION) -> List[float]:
    """
    Fallback determinístico quando embeddings falham.

    Mantém consistência:
    - testes offline
    - falhas de API
    - fallback resiliente
    """

    seed = int(sha256_hash(text.encode()), 16) % (2**32)

    rng = random.Random(seed)

    return [rng.random() for _ in range(dim)]


# ---------------------------------------------------------
# Adaptive Embedding Router
# ---------------------------------------------------------

class AdaptiveEmbeddingRouter:
    """
    Router adaptativo de embeddings.

    Resolve automaticamente:

    - provider via container
    - dimensão correta
    - embed_batch se disponível
    - fallback resiliente
    """

    def __init__(self):

        self._provider = None
        self._dimension = DEFAULT_DIMENSION
        self._batch_supported = False

    # --------------------------------------------------

    def _load_provider(self):

        try:
            provider = container.resolve("embedding_provider")
        except Exception:
            provider = None

        self._provider = provider

        if not provider:
            return

        # detectar dimensão
        self._dimension = (
            getattr(provider, "dimension", None)
            or getattr(provider, "dim", None)
            or DEFAULT_DIMENSION
        )

        # detectar batch
        batch_fn = getattr(provider, "embed_batch", None)

        self._batch_supported = callable(batch_fn)

    # --------------------------------------------------

    def _ensure_provider(self):

        if self._provider is None:
            self._load_provider()

        return self._provider

    # --------------------------------------------------

    async def _call_embed(self, provider, text):

        result = provider.embed(text)

        if inspect.isawaitable(result):
            result = await result

        return result

    # --------------------------------------------------
    # single embedding
    # --------------------------------------------------

    async def embed(self, text: str) -> List[float]:

        if not text.strip():
            return deterministic_vector(text, self._dimension)

        provider = self._ensure_provider()

        if not provider:
            return deterministic_vector(text, self._dimension)

        async def provider_call(t):

            return await self._call_embed(provider, t)

        return await resilient_call(
            [
                provider_call,
                lambda t: deterministic_vector(t, self._dimension)
            ],
            text,
            speculative_delay=0.2
        )

    # --------------------------------------------------
    # batch embedding
    # --------------------------------------------------

    async def embed_batch(self, texts: Iterable[str]) -> List[List[float]]:

        texts = list(texts)

        if not texts:
            return []

        provider = self._ensure_provider()

        if not provider:

            return [
                deterministic_vector(t, self._dimension)
                for t in texts
            ]

        if self._batch_supported:

            batch_fn = provider.embed_batch

            try:

                result = batch_fn(texts)

                if inspect.isawaitable(result):
                    result = await result

                return result

            except Exception:
                pass

        # fallback sequential

        vectors = []

        for t in texts:
            vec = await self.embed(t)
            vectors.append(vec)

        return vectors


# ---------------------------------------------------------
# singleton router
# ---------------------------------------------------------

_router = AdaptiveEmbeddingRouter()


# ---------------------------------------------------------
# public API
# ---------------------------------------------------------

async def remote_embed(text: str) -> List[float]:
    return await _router.embed(text)


async def remote_embed_batch(texts: Iterable[str]) -> List[List[float]]:
    return await _router.embed_batch(texts)
