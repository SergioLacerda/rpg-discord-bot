import time
import asyncio
from collections import OrderedDict

from rpgbot.core.runtime_state import get_event_version
from rpgbot.infrastructure.vector_index import VectorIndex
from rpgbot.utils.text.query_expansion import expand_query
from rpgbot.infrastructure.embedding_cache import embed
from rpgbot.utils.vector.vector_utils import project


class RetrievalEngine:

    def __init__(
        self,
        index: VectorIndex | None = None,
        cache_size=128,
        embed_cache_size=256,
        ttl_seconds=300
    ):

        self.index = index or VectorIndex()

        self.cache_size = cache_size
        self.embed_cache_size = embed_cache_size

        self.cache = OrderedDict()
        self.embed_cache = OrderedDict()

        self._lock = asyncio.Lock()
        self._inflight = {}

        self.ttl = ttl_seconds

        self.event_version = get_event_version()

    # ---------------------------------------------------------
    # TTL
    # ---------------------------------------------------------

    def _valid(self, ts):
        return (time.time() - ts) < self.ttl

    # ---------------------------------------------------------
    # Invalidation
    # ---------------------------------------------------------

    def _check_invalidation(self):

        current_version = get_event_version()

        if current_version != self.event_version:

            self.cache.clear()
            self.embed_cache.clear()

            self.event_version = current_version

    # ---------------------------------------------------------
    # Semantic cluster key
    # ---------------------------------------------------------

    def _semantic_key(self, query_vec, k):

        proj = project(query_vec)

        bucket = int(proj * 100)  # cluster semântico

        return (bucket, k)

    # ---------------------------------------------------------
    # Embedding cache
    # ---------------------------------------------------------

    async def get_embedding(self, query):

        self._check_invalidation()

        entry = self.embed_cache.get(query)

        if entry and self._valid(entry["ts"]):

            self.embed_cache.move_to_end(query)

            return entry["vec"]

        expanded_query = expand_query(query)

        vec = await embed(expanded_query)

        async with self._lock:

            self.embed_cache[query] = {
                "vec": vec,
                "ts": time.time()
            }

            if len(self.embed_cache) > self.embed_cache_size:
                self.embed_cache.popitem(last=False)

        return vec

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    async def search(self, query, k=4):

        self._check_invalidation()

        q_vec = await self.get_embedding(query)

        key = self._semantic_key(q_vec, k)

        entry = self.cache.get(key)

        if entry and self._valid(entry["ts"]):

            self.cache.move_to_end(key)

            return entry["result"]

        # inflight deduplication

        task = self._inflight.get(key)

        if task:
            return await task

        async def _execute():

            return await self.index.search(query, q_vec, k)

        task = asyncio.create_task(_execute())

        self._inflight[key] = task

        try:

            result = await task

        finally:

            self._inflight.pop(key, None)

        async with self._lock:

            self.cache[key] = {
                "result": result,
                "ts": time.time()
            }

            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

        return result