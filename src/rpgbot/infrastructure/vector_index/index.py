import asyncio
from collections import defaultdict

from rpgbot.core.paths import CAMPAIGN_DIR
from rpgbot.utils.text.normalize_utils import tokenize

from .pipeline.context import SearchContext
from .stages.query_local_cache import QueryLocalCache

from .pipeline.retrieval_pipeline import RetrievalPipeline
from .retrieval.candidate_retriever import CandidateRetriever
from .retrieval.routing import EntityRecallRouter
from .expansion.temporal_expansion import TemporalExpansion
from .expansion.causal_expansion import CausalExpansion


class VectorIndex:

    def __init__(
        self,
        components,
        semantic_cache,
        document_loader,
        repository,
        embedding_indexer,
        campaign_dir=CAMPAIGN_DIR,
        campaign_id="default",
        temporal_index=None
    ):

        self.components = components

        self.query_classifier = components.query_classifier
        self.stage1_ranker = components.stage1_ranker
        self.stage2_ranker = components.stage2_ranker
        self.cluster_manager = components.cluster_manager

        self.vector_store = components.vector_store
        self.document_store = components.document_store
        self.token_store = components.token_store
        self.metadata_store = components.metadata_store

        self.ivf_builder = components.ivf_builder
        self.ivf_router = components.ivf_router

        self.semantic_cache = semantic_cache
        self.document_loader = document_loader
        self.repository = repository
        self.embedding_indexer = embedding_indexer

        self.campaign_dir = campaign_dir
        self.campaign_id = campaign_id

        self.temporal_index = temporal_index

        # ---------------------------------------------------------
        # runtime state
        # ---------------------------------------------------------

        self.doc_ids = []
        self.entity_graph = defaultdict(set)

        self.projections = None

        self.bm25_index = None

        self.query_cache = QueryLocalCache()

        self._loaded = False
        self._ann_ready = False

        self._load_lock = asyncio.Lock()
        self._ann_lock = asyncio.Lock()

        # caches
        self.query_vector_cache = {}
        self.result_cache = {}
        self.cache_limit = 512

        # retrieval budgets
        self.vector_budget = 60
        self.lexical_budget = 40
        self.graph_budget = 20
        self.max_candidates = 120

        # pipeline será criado após load
        self.retrieval_pipeline = None

    # ---------------------------------------------------------
    # PIPELINE BUILDER
    # ---------------------------------------------------------

        def _build_pipeline(self):

            retriever = CandidateRetriever(
                docs=self.doc_ids,
                projections=self.projections,
                vector_store=self.vector_store,
                ivf_router=self.ivf_router
            )

            entity_router = EntityRecallRouter(
                docs=self.doc_ids,
                entity_resolver=lambda q: q.split()
            )

            temporal = TemporalExpansion()

            causal = CausalExpansion(
                causality_graph=self.entity_graph,
                doc_lookup=self.document_store
            )

            self.retrieval_pipeline = RetrievalPipeline(
                recall=[retriever, entity_router],
                expansion=[temporal, causal],
                filtering=[]
            )

    # ---------------------------------------------------------
    # Lazy ANN
    # ---------------------------------------------------------

    async def ensure_ann_ready(self):

        if self._ann_ready:
            return

        async with self._ann_lock:

            if self._ann_ready:
                return

            if not self.doc_ids:
                return

            self.ivf_index = self.ivf_builder.build(
                self.doc_ids,
                self.vector_store
            )

            self.ivf_router.set_index(self.ivf_index)

            self._ann_ready = True

    # ---------------------------------------------------------
    # Load index (incremental)
    # ---------------------------------------------------------

    async def load(self):

        if self._loaded:
            return

        async with self._load_lock:

            if self._loaded:
                return

            campaign_path = self.campaign_dir / self.campaign_id

            raw_docs = self.document_loader.discover(campaign_path)

            persisted = self.repository.load()

            docs, changed = await self.embedding_indexer.build_incremental(
                raw_docs,
                persisted
            )

            if not docs:
                raise RuntimeError("VectorIndex carregado sem documentos")

            # reset state
            self.doc_ids.clear()
            self.entity_graph.clear()

            self.vector_store.clear()
            self.document_store.clear()
            self.token_store.clear()
            self.metadata_store.clear()

            if self.temporal_index:
                self.temporal_index.clear()

            projections = []

            append = self.doc_ids.append
            add_vec = self.vector_store.add
            add_doc = self.document_store.add
            add_tok = self.token_store.add
            add_meta = self.metadata_store.add

            entity_graph = self.entity_graph

            for doc in docs:

                doc_id = doc["id"]

                append(doc_id)

                add_vec(doc_id, doc["vector"])

                add_doc(doc_id, doc["text"], doc["source"])

                tokens = doc["tokens"]
                token_set = doc["token_set"]

                add_tok(doc_id, tokens, token_set)

                ts = doc.get("timestamp")

                add_meta(
                    doc_id,
                    timestamp=ts,
                    mtime=doc.get("mtime")
                )

                if self.temporal_index and ts:
                    self.temporal_index.add(doc_id, ts)

                for t in token_set:
                    if len(t) > 3 and t[0].isalpha():
                        entity_graph[t].add(doc_id)

                proj = doc.get("projection")
                if proj is not None:
                    projections.append(proj)

            self.projections = sorted(projections) if projections else None

            if changed:

                self.cluster_manager.update(self.doc_ids, self.vector_store)

                self._ann_ready = False

            if self.bm25_index and changed:
                self.bm25_index.build(self.doc_ids, self.token_store)

            self.repository.save(docs)

            # pipeline agora que temos docs
            self._build_pipeline()

            self._loaded = True
# ---------------------------------------------------------
# Graph candidates
# ---------------------------------------------------------

def _graph_candidates(self, query_tokens):

    result = set()

    graph = self.entity_graph

    for t in set(query_tokens):

        docs = graph.get(t)

        if docs:
            result.update(docs)

        if len(result) >= self.graph_budget * 5:
            break

    return list(result)


# ---------------------------------------------------------
# Hybrid fusion
# ---------------------------------------------------------

def _fusion(self, vector_ids, lexical_ids, graph_ids, limit=None):

    limit = limit or self.max_candidates

    scores = {}
    k = 60

    for rank, doc_id in enumerate(vector_ids[:self.vector_budget]):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    for rank, doc_id in enumerate(lexical_ids[:self.lexical_budget]):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    for rank, doc_id in enumerate(graph_ids[:self.graph_budget]):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_id for doc_id, _ in ranked[:limit]]


# ---------------------------------------------------------
# Search
# ---------------------------------------------------------

async def search(self, query, q_vec=None, k=4):

    if not query:
        return []

    if not self.retrieval_pipeline:
        self._build_pipeline()

    query = query.strip().lower()

    if not query:
        return []

    # ----------------------------------------
    # fast result cache
    # ----------------------------------------

    cached = self.result_cache.get(query)

    if cached:
        return cached

    # ----------------------------------------
    # vector cache
    # ----------------------------------------

    if q_vec is None:

        q_vec = self.query_vector_cache.get(query)

        if q_vec is None:

            q_vec = await self.embedding_indexer.embed(query)

            if len(self.query_vector_cache) > self.cache_limit:
                self.query_vector_cache.clear()

            self.query_vector_cache[query] = q_vec

    # ----------------------------------------
    # ensure index loaded
    # ----------------------------------------

    if not self._loaded:
        await self.load()

    if not self.retrieval_pipeline:
        self._build_pipeline()

    await self.ensure_ann_ready()

    # ----------------------------------------
    # preprocess query
    # ----------------------------------------

    final_query, q_vec = await self.query_preprocessor.prepare(query, q_vec)

    # ----------------------------------------
    # semantic cache
    # ----------------------------------------

    if len(final_query) > 4:

        cached = self.semantic_cache.get(final_query)

        if cached:

            get_doc = self.document_store.get

            return [
                get_doc(doc_id)["text"]
                for doc_id in cached
                if get_doc(doc_id)
            ]

    # ----------------------------------------
    # tokenize
    # ----------------------------------------

    query_tokens = tokenize(final_query)

    # ----------------------------------------
    # entity prefilter
    # ----------------------------------------

    prefilter_ids = self._entity_prefilter(query_tokens)

    query_type = self.query_classifier.classify(final_query)

    ctx = SearchContext(
        query=final_query,
        q_vec=q_vec,
        query_tokens=query_tokens,
        query_type=query_type,
        vector_store=self.vector_store,
        token_store=self.token_store,
        metadata_store=self.metadata_store,
        cluster_manager=self.cluster_manager,
        ivf_router=self.ivf_router,
        temporal_index=self.temporal_index,
        prefilter_ids=prefilter_ids
    )

    ctx.index = self

    # ----------------------------------------
    # vector retrieval
    # ----------------------------------------

    cached_candidates = self.query_cache.run(ctx, [])

    if cached_candidates:

        vector_candidates = cached_candidates

    else:

        vector_candidates = await self.retrieval_pipeline.run(ctx)

        vector_candidates = vector_candidates[: self.vector_budget]

        self.query_cache.update(ctx, vector_candidates)

    # ----------------------------------------
    # parallel retrieval
    # ----------------------------------------

    bm25_task = asyncio.create_task(self._bm25(query_tokens))
    graph_task = asyncio.create_task(self._graph_candidates(query_tokens))

    lexical_candidates, graph_candidates = await asyncio.gather(
        bm25_task,
        graph_task
    )

    # ----------------------------------------
    # fusion
    # ----------------------------------------

    candidate_ids = self._fusion(
        vector_candidates,
        lexical_candidates,
        graph_candidates
    )

    if not candidate_ids:
        return []

    # ----------------------------------------
    # stage1 ranking
    # ----------------------------------------

    stage1 = self.stage1_ranker.rank(
        q_vec,
        query_tokens,
        candidate_ids,
        self.vector_store,
        self.token_store,
        k
    )

    if not stage1:
        return []

    # ----------------------------------------
    # stage2 ranking
    # ----------------------------------------

    top_ids = self.stage2_ranker.rank(
        stage1,
        query_tokens,
        self.vector_store,
        self.token_store,
        self.metadata_store,
        k
    )

    if not top_ids:
        return []

    # ----------------------------------------
    # semantic cache update
    # ----------------------------------------

    if len(final_query) > 4:
        self.semantic_cache.set(final_query, top_ids)

    # ----------------------------------------
    # fetch docs
    # ----------------------------------------

    get_doc = self.document_store.get

    result = []

    for doc_id in top_ids:

        doc = get_doc(doc_id)

        if doc:
            result.append(doc["text"])

    # ----------------------------------------
    # result cache
    # ----------------------------------------

    if len(self.result_cache) > self.cache_limit:
        self.result_cache.clear()

    self.result_cache[query] = result

    return result