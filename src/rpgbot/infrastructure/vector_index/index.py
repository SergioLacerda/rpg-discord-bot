import asyncio
from collections import defaultdict

from rpgbot.core.config.paths import CAMPAIGN_DIR
from rpgbot.utils.text.normalize_utils import tokenize

from .indexing.narrative_timeline_index import NarrativeTimelineIndex
from .indexing.narrative_causality_graph import NarrativeCausalityGraph
from .expansion.temporal_expansion import TemporalExpansion
from .expansion.causal_expansion import CausalExpansion
from .expansion.timeline_expansion import TimelineExpansion
from .pipeline.context import SearchContext
from .pipeline.retrieval_pipeline import RetrievalPipeline

from .retrieval.candidate_retriever import CandidateRetriever
from .retrieval.routing import EntityRecallRouter

from .stages.entity_memory import NarrativeEntityMemory
from .stages.narrative_memory import NarrativeMemoryStage
from .stages.narrative_window import NarrativeWindowRetriever
from .stages.query_local_cache import QueryLocalCache

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

        # ---------------------------------------------------------
        # components
        # ---------------------------------------------------------

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


        # ---------------------------------------------------------
        # external services
        # ---------------------------------------------------------

        self.semantic_cache = semantic_cache
        self.document_loader = document_loader
        self.repository = repository
        self.embedding_indexer = embedding_indexer


        # ---------------------------------------------------------
        # campaign
        # ---------------------------------------------------------

        self.campaign_dir = campaign_dir
        self.campaign_id = campaign_id

        self.temporal_index = temporal_index


        # ---------------------------------------------------------
        # runtime state
        # ---------------------------------------------------------

        self.doc_ids = []
        self.doc_positions = {}

        self.entity_graph = defaultdict(set)
        self.entity_memory = {}

        self.causality_graph = NarrativeCausalityGraph()
        self.importance_store = {}

        self.projections = None

        self.bm25_index = None

        self.query_cache = QueryLocalCache()


        # ---------------------------------------------------------
        # async state
        # ---------------------------------------------------------

        self._loaded = False
        self._ann_ready = False

        self._load_lock = asyncio.Lock()
        self._ann_lock = asyncio.Lock()


        # ---------------------------------------------------------
        # caches
        # ---------------------------------------------------------

        self.query_vector_cache = {}
        self.result_cache = {}

        self.cache_limit = 512


        # ---------------------------------------------------------
        # retrieval budgets
        # ---------------------------------------------------------

        self.vector_budget = 60
        self.lexical_budget = 40
        self.graph_budget = 20
        self.max_candidates = 120


        # ---------------------------------------------------------
        # retrieval tuning
        # ---------------------------------------------------------

        self.retrieval_config = {

            # causal expansion
            "causal_depth": 2,
            "causal_limit": 20,
            "causal_per_doc": 4,

            # narrative window
            "window_before": 2,
            "window_after": 2,

            # entity recall
            "entity_min_docs": 2,

            # importance scoring
            "importance_weight": 0.15
        }


        self.timeline_index = NarrativeTimelineIndex()

        # ---------------------------------------------------------
        # pipeline (built after load)
        # ---------------------------------------------------------

        self.retrieval_pipeline = None

    # ---------------------------------------------------------
    # PIPELINE BUILDER
    # ---------------------------------------------------------

    def _build_pipeline(self):

        # ---------------------------------------------------------
        # candidate retriever
        # ---------------------------------------------------------

        retriever = CandidateRetriever(
            docs=self.doc_ids,
            projections=self.projections,
            vector_store=self.vector_store,
            ivf_router=self.ivf_router
        )

        retriever.priority = 20


        # ---------------------------------------------------------
        # narrative memory layers
        # ---------------------------------------------------------

        memory_layers = []

        for name in (
            "world_lore_index",
            "arc_index",
            "session_index",
            "recent_index",
        ):

            layer = getattr(self, name, None)

            if layer:
                memory_layers.append(layer)

        hierarchical = []

        if memory_layers:
            hierarchical.append(NarrativeMemoryStage(memory_layers))


        # ---------------------------------------------------------
        # Narrative entity memory
        # ---------------------------------------------------------

        entity_memory = None

        if getattr(self, "entity_memory", None):

            entity_memory = NarrativeEntityMemory(self.entity_memory)
            entity_memory.priority = 30


        # ---------------------------------------------------------
        # entity router
        # ---------------------------------------------------------

        entity_router = EntityRecallRouter(
            docs=self.doc_ids,
            entity_resolver=lambda q: q.split()
        )

        entity_router.priority = 35


        # ---------------------------------------------------------
        # Narrative window
        # ---------------------------------------------------------

        window = NarrativeWindowRetriever(
            docs=self.doc_ids,
            window_before=2,
            window_after=2
        )

        window.priority = 40


        # ---------------------------------------------------------
        # expansions
        # ---------------------------------------------------------

        temporal = TemporalExpansion()
        temporal.priority = 45

        timeline = TimelineExpansion()
        timeline.priority = 47

        causal = CausalExpansion(
            max_expansion=20,
            depth=2,
            per_doc_limit=4
        )

        causal.priority = 50


        # ---------------------------------------------------------
        # Narrative importance
        # ---------------------------------------------------------

        importance_stage = None

        if getattr(self, "importance_store", None):

            importance_stage = NarrativeImportanceStage(self.importance_store)
            importance_stage.priority = 60


        # ---------------------------------------------------------
        # assemble pipeline lists
        # ---------------------------------------------------------

        recall = [retriever]

        if entity_memory:
            recall.append(entity_memory)

        recall.append(entity_router)

        expansion = [
            window,
            temporal,
            timeline,
            causal
        ]

        filtering = []

        if importance_stage:
            filtering.append(importance_stage)


        # ---------------------------------------------------------
        # build pipeline
        # ---------------------------------------------------------

        self.retrieval_pipeline = RetrievalPipeline(
            hierarchical=hierarchical,
            recall=recall,
            expansion=expansion,
            filtering=filtering
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

            # ---------------------------------------------------------
            # Batch embeddings (se necessário)
            # ---------------------------------------------------------

            texts = []
            missing_idx = []

            for i, doc in enumerate(docs):

                if "vector" not in doc or doc["vector"] is None:

                    texts.append(doc["text"])
                    missing_idx.append(i)

            if texts:

                embedder = self.embedding_indexer

                if hasattr(embedder, "embed_batch"):

                    vectors = await embedder.embed_batch(texts)

                else:

                    vectors = []

                    for t in texts:
                        vectors.append(await embedder.embed(t))

                for idx, vec in zip(missing_idx, vectors):
                    docs[idx]["vector"] = vec

            prev_doc_id = None
            doc_positions = self.doc_positions
            importance_store = self.importance_store
            timeline_index = self.timeline_index
            causality_graph = self.causality_graph

            for pos, doc in enumerate(docs):

                doc_id = doc["id"]
                text = doc["text"]
                source = doc["source"]

                vector = doc["vector"]

                tokens = doc["tokens"]
                token_set = doc["token_set"]

                ts = doc.get("timestamp")
                mtime = doc.get("mtime")

                proj = doc.get("projection")

                # ---------------------------------------------------------
                # doc ids
                # ---------------------------------------------------------

                append(doc_id)
                doc_positions[doc_id] = pos

                # ---------------------------------------------------------
                # vector store
                # ---------------------------------------------------------

                add_vec(doc_id, vector)

                # ---------------------------------------------------------
                # document store
                # ---------------------------------------------------------

                add_doc(doc_id, text, source)

                # ---------------------------------------------------------
                # token store
                # ---------------------------------------------------------

                add_tok(doc_id, tokens, token_set)

                # ---------------------------------------------------------
                # metadata
                # ---------------------------------------------------------

                add_meta(
                    doc_id,
                    timestamp=ts,
                    mtime=mtime
                )

                # ---------------------------------------------------------
                # timeline index
                # ---------------------------------------------------------

                if ts:
                    timeline_index.add(doc_id, ts)

                # ---------------------------------------------------------
                # temporal index
                # ---------------------------------------------------------

                if self.temporal_index and ts:
                    self.temporal_index.add(doc_id, ts)

                # ---------------------------------------------------------
                # causal graph
                # ---------------------------------------------------------

                if prev_doc_id and ts:
                    causality_graph.add_edge(prev_doc_id, doc_id)

                prev_doc_id = doc_id

                # ---------------------------------------------------------
                # entity graph
                # ---------------------------------------------------------

                for t in token_set:

                    if len(t) > 3 and t[0].isalpha():
                        entity_graph[t].add(doc_id)

                # ---------------------------------------------------------
                # narrative importance
                # ---------------------------------------------------------

                importance = 1

                text_lower = text.lower()

                if "morto" in text_lower or "killed" in text_lower:
                    importance = 5
                elif "descobriu" in text_lower or "found" in text_lower:
                    importance = 4

                importance_store[doc_id] = importance

                # ---------------------------------------------------------
                # projections
                # ---------------------------------------------------------

                if proj is not None:
                    projections.append(proj)

            # ---------------------------------------------------------
            # Narrative Entity Memory (auto build)
            # ---------------------------------------------------------

            entity_memory = {}

            for token, docs_set in entity_graph.items():

                # evitar tokens muito raros
                if len(docs_set) < 2:
                    continue

                entity_memory[token] = {
                    "aliases": [],
                    "docs": set(docs_set),
                    "type": "unknown"
                }

            self.entity_memory = entity_memory

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