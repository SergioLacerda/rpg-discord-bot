import time
import asyncio
from collections import OrderedDict

from rpgbot.campaign.memory.entity_alias_resolver import EntityAliasResolver
from rpgbot.core.runtime import SEARCH_EXECUTOR
from rpgbot.core.runtime_state import get_event_version
from rpgbot.core.container import container
from rpgbot.infrastructure.vector_index import VectorIndex
from rpgbot.infrastructure.embedding_cache import embed
from rpgbot.utils.text.query_expansion import expand_query
from rpgbot.utils.vector.vector_utils import cosine_similarity


class RetrievalEngine:

    def __init__(
        self,
        index: VectorIndex | None = None,
        cache_size=128,
        embed_cache_size=256,
        ttl_seconds=300
    ):

        # multi-campaign index registry
        self.indexes = {}
        self.default_index = index or container.resolve("vector_index")

        self.cache_size = cache_size
        self.embed_cache_size = embed_cache_size

        self.cache = OrderedDict()
        self.embed_cache = OrderedDict()

        self._lock = asyncio.Lock()
        self._inflight = {}

        self.ttl = ttl_seconds

        self.event_version = get_event_version()
        self.alias_resolver = EntityAliasResolver()

        self.query_memory = OrderedDict()
        self.memory_size = 64

        # async batching
        self._batch = []
        self._batch_task = None
        self._batch_window = 0.008

    # ---------------------------------------------------------
    # Campaign index resolver
    # ---------------------------------------------------------

    def _get_index(self, campaign_id):

        if campaign_id is None:
            return self.default_index

        index = self.indexes.get(campaign_id)

        if index:
            return index

        # lazy load campaign index
        index = container.resolve("vector_index_factory")(campaign_id)

        self.indexes[campaign_id] = index

        return index

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
            self.query_memory.clear()

            self.event_version = current_version

    # ---------------------------------------------------------
    # Semantic bucket
    # ---------------------------------------------------------

    def _semantic_key(self, query_vec, query, k):

        bucket = tuple(int(v * 8) for v in query_vec[:4])
        length_bucket = len(query) // 5

        return (bucket, length_bucket, k)

    # ---------------------------------------------------------
    # Semantic memory
    # ---------------------------------------------------------

    def _semantic_memory_lookup(self, query_vec):

        best_score = 0
        best_result = None

        for entry in self.query_memory.values():

            if not self._valid(entry["ts"]):
                continue

            score = cosine_similarity(query_vec, entry["vec"])

            if score > best_score:
                best_score = score
                best_result = entry["result"]

        if best_score > 0.92:
            return best_result

        return None

    # ---------------------------------------------------------
    # Embedding cache
    # ---------------------------------------------------------

    async def get_embedding(self, query):

        self._check_invalidation()

        async with self._lock:

            entry = self.embed_cache.get(query)

            if entry and self._valid(entry["ts"]):

                self.embed_cache.move_to_end(query)

                return entry["vec"]

        expanded_query = expand_query(query)
        final_query = self.alias_resolver.normalize(expanded_query)

        vec = await embed(final_query)

        async with self._lock:

            self.embed_cache[query] = {
                "vec": vec,
                "ts": time.time()
            }

            if len(self.embed_cache) > self.embed_cache_size:
                self.embed_cache.popitem(last=False)

        return vec

    # ---------------------------------------------------------
    # Threaded index search
    # ---------------------------------------------------------

    async def _run_index_search(self, index, query, q_vec, k):

        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            SEARCH_EXECUTOR,
            lambda: asyncio.run(
                index.search(query, q_vec, k)
            )
        )

    # ---------------------------------------------------------
    # Internal search
    # ---------------------------------------------------------

    async def _search_internal(self, query, k, campaign_id):

        index = self._get_index(campaign_id)

        q_vec = await self.get_embedding(query)

        memory_hit = self._semantic_memory_lookup(q_vec)

        if memory_hit:
            return memory_hit

        key = self._semantic_key(q_vec, query, k)

        async with self._lock:

            entry = self.cache.get(key)

            if entry and self._valid(entry["ts"]):

                self.cache.move_to_end(key)

                return entry["result"]

        task = self._inflight.get(key)

        if task:
            return await task

        async def _execute():
            return await self._run_index_search(index, query, q_vec, k)

        task = asyncio.create_task(_execute())

        self._inflight[key] = task

        try:
            result = await task
        finally:
            self._inflight.pop(key, None)

        if len(query) > 6:

            self.query_memory[query] = {
                "vec": q_vec,
                "result": result,
                "ts": time.time()
            }

            if len(self.query_memory) > self.memory_size:
                self.query_memory.popitem(last=False)

        async with self._lock:

            self.cache[key] = {
                "result": result,
                "ts": time.time()
            }

            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

        return result

    # ---------------------------------------------------------
    # Batch processor
    # ---------------------------------------------------------

    async def _process_batch(self):

        await asyncio.sleep(self._batch_window)

        batch = self._batch
        self._batch = []
        self._batch_task = None

        for fut, query, k, campaign_id in batch:

            try:

                result = await self._search_internal(
                    query,
                    k,
                    campaign_id
                )

                fut.set_result(result)

            except Exception as e:
                fut.set_exception(e)

    # ---------------------------------------------------------
    # Public search
    # ---------------------------------------------------------

    async def search(self, query, k=4, campaign_id=None):

        loop = asyncio.get_running_loop()

        fut = loop.create_future()

        self._batch.append((fut, query, k, campaign_id))

        if not self._batch_task:
            self._batch_task = asyncio.create_task(self._process_batch())

        return await fut