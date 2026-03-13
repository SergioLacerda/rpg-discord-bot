import bisect
import hashlib
import heapq
import random
import re
import time
from pathlib import Path
from collections import defaultdict

from rpgbot.infrastructure.narrative_graph import related_entities
from rpgbot.utils.vector.vector_utils import (
    cosine_similarity,
    lsh_hash,
    project,
    keyword_score,
)
from rpgbot.utils.text.normalize_utils import tokenize
from rpgbot.utils.text.query_expansion import expand_query
from rpgbot.utils.text.ranking_utils import entity_boost, contextual_score
from rpgbot.utils import load_json, save_json
from rpgbot.rag.hnsw_index import HNSWIndex


VECTOR_FILE = Path("campaign/index_vectors.json")
EVENT_FILE = Path("campaign/memory/events.json")


class VectorIndex:

    def __init__(
        self,
        embed,
        alias_resolver,
        semantic_cache,
        ann_index_factory,
        vector_file=VECTOR_FILE,
        campaign_dir=Path("campaign"),
        campaign_id="default"
    ):

        self.embed = embed
        self.alias_resolver = alias_resolver
        self.semantic_cache = semantic_cache
        self.ann_index_factory = ann_index_factory

        self.campaign_id = campaign_id
        self.campaign_dir = campaign_dir
        self.vector_file = campaign_dir / campaign_id / "index_vectors.json"

        self.docs = []
        self.doc_lookup = {}

        self.projections = []
        self.lsh_buckets = {}

        self.inverted_index = {}

        self.centroids = []
        self.centroid_projections = []
        self.cluster_docs = {}

        self.cluster_count = 16
        self.super_centroids = []
        self.super_clusters = {}
        self.super_cluster_count = 8

        self.hnsw_index = None
        self.ann_index = None

        self.entity_graph = {}
        self.doc_entities = {}
        self.causality_graph = {}

        # drift protection
        self._last_cluster_size = 0
        self._cluster_rebuild_threshold = 0.20

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------

    @staticmethod
    def _hash_text(text):
        return hashlib.sha1(text.encode()).hexdigest()

    # ---------------------------------------------------------
    # adaptive cluster sizing
    # ---------------------------------------------------------

    def _adaptive_cluster_sizes(self):

        n = len(self.docs)

        if n < 50:
            self.cluster_count = 4
            self.super_cluster_count = 2
            return

        cluster_target = int(n ** 0.5)
        cluster_target = max(8, min(cluster_target, 512))

        super_target = int(cluster_target ** 0.5)
        super_target = max(4, min(super_target, 64))

        self.cluster_count = cluster_target
        self.super_cluster_count = super_target

    # ---------------------------------------------------------
    # clustering
    # ---------------------------------------------------------

    def _build_clusters(self):

        if not self.docs:
            return

        vectors = [d["vector"] for d in self.docs]

        self.centroids = random.sample(
            vectors,
            min(self.cluster_count, len(vectors))
        )

        # kmeans-lite refinement

        for _ in range(2):

            cluster_docs = {i: [] for i in range(len(self.centroids))}

            for doc in self.docs:

                best = 0
                best_score = -1

                for i, c in enumerate(self.centroids):

                    s = cosine_similarity(doc["vector"], c)

                    if s > best_score:
                        best_score = s
                        best = i

                cluster_docs[best].append(doc)

            new_centroids = []

            for i in range(len(self.centroids)):

                docs = cluster_docs[i]

                if not docs:
                    new_centroids.append(self.centroids[i])
                    continue

                dim = len(docs[0]["vector"])
                mean = [0.0] * dim

                for d in docs:
                    vec = d["vector"]
                    for j in range(dim):
                        mean[j] += vec[j]

                mean = [v / len(docs) for v in mean]

                new_centroids.append(mean)

            self.centroids = new_centroids
            self.cluster_docs = cluster_docs

        # projection routing

        self.centroid_projections = [
            project(c) for c in self.centroids
        ]

        pairs = sorted(
            zip(self.centroid_projections, self.centroids),
            key=lambda x: x[0]
        )

        self.centroid_projections = [p for p, _ in pairs]
        self.centroids = [c for _, c in pairs]

        self._last_cluster_size = len(self.docs)

    # ---------------------------------------------------------
    # drift protection
    # ---------------------------------------------------------

    def _should_rebuild_clusters(self):

        if not self.centroids:
            return True

        if self._last_cluster_size == 0:
            return True

        growth = abs(len(self.docs) - self._last_cluster_size)

        return growth / self._last_cluster_size > self._cluster_rebuild_threshold

    # ---------------------------------------------------------
    # projection routing
    # ---------------------------------------------------------

    def projection_routing(self, q_vec, top_k=20):

        if not self.centroid_projections:
            return []

        q_proj = project(q_vec)

        pos = bisect.bisect_left(self.centroid_projections, q_proj)

        window = max(5, top_k)

        start = max(0, pos - window)
        end = min(len(self.centroid_projections), pos + window)

        return list(range(start, end))

    # ---------------------------------------------------------
    # cluster routing
    # ---------------------------------------------------------

    def cluster_candidates(self, q_vec, top_clusters=3):

        candidate_ids = self.projection_routing(q_vec)

        scored = []

        for i in candidate_ids:

            centroid = self.centroids[i]

            s = cosine_similarity(q_vec, centroid)

            scored.append((s, i))

        scored.sort(reverse=True)

        docs = []

        for _, cid in scored[:top_clusters]:
            docs.extend(self.cluster_docs.get(cid, []))

        return docs


    def route_retrieval(self, query_type, candidates):

        if query_type == "lore":

            # lore tende a estar em documentos antigos
            return sorted(
                candidates,
                key=lambda d: d.get("timestamp", 0)
            )

        if query_type == "memory":

            # memória recente
            return sorted(
                candidates,
                key=lambda d: d.get("timestamp", 0),
                reverse=True
            )

        if query_type == "investigation":

            # expandir causalidade
            expanded = self.causal_candidates(candidates)

            if expanded:
                candidates.extend(expanded)

        return candidates


    def classify_query(self, query: str) -> str:

        q = query.lower()

        lore_patterns = (
            "quem é",
            "quem foi",
            "o que é",
            "quem são",
            "who is",
            "what is"
        )

        memory_patterns = (
            "o que aconteceu",
            "o que houve",
            "quando aconteceu",
            "recentemente",
            "what happened",
            "recent"
        )

        investigation_patterns = (
            "por que",
            "quem causou",
            "como aconteceu",
            "o que levou",
            "why",
            "what caused"
        )

        if any(p in q for p in investigation_patterns):
            return "investigation"

        if any(p in q for p in lore_patterns):
            return "lore"

        if any(p in q for p in memory_patterns):
            return "memory"

        return "semantic"


    # ---------------------------------------------------------
    # indexing
    # ---------------------------------------------------------

    def load(self):

        campaign_path = self.campaign_dir / self.campaign_id

        if not campaign_path.exists():
            self.docs = []
            return

        persisted = load_json(self.vector_file, [])

        existing = {
            d["source"]: d
            for d in persisted
            if d.get("source")
        }

        updated_docs = []

        for file in campaign_path.glob("**/*.md"):

            source = str(file)

            text = file.read_text(encoding="utf-8")
            mtime = file.stat().st_mtime

            text_hash = self._hash_text(text)

            prev = existing.get(source)

            if prev and prev.get("hash") == text_hash:
                updated_docs.append(prev)
                continue

            vec = self.embed(text)

            updated_docs.append({
                "text": text,
                "vector": vec,
                "source": source,
                "mtime": mtime,
                "hash": text_hash,
                "timestamp": mtime
            })

        self.docs = updated_docs
        self.doc_lookup = {id(d): i for i, d in enumerate(self.docs)}

        # rebuild clusters only if needed

        if self._should_rebuild_clusters():

            self._adaptive_cluster_sizes()
            self._build_clusters()

        save_json(self.vector_file, self.docs)


    # ---------------------------------------------------------
    # search
    # ---------------------------------------------------------

    async def search(self, query, q_vec=None, k=4):

        if len(query) < 4:
            return []

        if not self.docs:
            self.load()

        if q_vec is None:

            expanded_query = expand_query(query)
            final_query = self.alias_resolver.normalize(expanded_query)

            q_vec = await self.embed(final_query)

        else:
            final_query = query

        cached = self.semantic_cache.get(final_query, q_vec)

        if cached:
            return cached

        query_tokens = tokenize(final_query)
        query_type = self.classify_query(final_query)

        # -------------------------
        # coarse retrieval
        # -------------------------

        candidates = self.hierarchical_candidates(q_vec)

        if not candidates:
            candidates = list(self.lsh_buckets.get(lsh_hash(q_vec), []))

        if not candidates and self.hnsw_index:
            candidates = self.hnsw_index.search(q_vec, k=200)

        if not candidates:

            pos = bisect.bisect_left(self.projections, project(q_vec))

            start = max(0, pos - 50)
            end = min(len(self.docs), pos + 50)

            candidates = self.docs[start:end]

        # -------------------------
        # entity recall
        # -------------------------

        related = {e.lower() for e in related_entities(final_query)}

        if related:

            seen = {id(d) for d in candidates}

            for d in self.docs[:200]:

                if d["token_set"] & related and id(d) not in seen:
                    candidates.append(d)
                    seen.add(id(d))

        # -------------------------
        # ANN prefilter
        # -------------------------

        if self.ann_index:

            ann_candidates = self.ann_index.search(q_vec)

            if ann_candidates and len(ann_candidates) > 10:

                ann_set = {id(d) for d in ann_candidates}

                candidates = [d for d in candidates if id(d) in ann_set]

        # -------------------------
        # projection prefilter
        # -------------------------

        q_proj = project(q_vec)

        threshold = max(0.15, 1 / (1 + len(self.docs) ** 0.5))

        filtered = [
            d for d in candidates
            if abs(d["proj"] - q_proj) < threshold
        ]

        if filtered:
            candidates = filtered

        # -------------------------
        # vector prefilter
        # -------------------------

        scored = []

        for d in candidates:

            vec_score = cosine_similarity(q_vec, d["vector"])

            if vec_score > 0.05:
                scored.append((vec_score, d))

        scored.sort(reverse=True)

        candidates = [d for _, d in scored[:80]]

        # -------------------------
        # lexical + graph
        # -------------------------

        lexical = self.lexical_candidates(query_tokens)
        graph_docs = self.graph_candidates(query_tokens)

        if lexical or graph_docs:

            candidates = self.reciprocal_rank_fusion(
                candidates[:100] + graph_docs[:100],
                lexical[:100] if lexical else [],
                limit=150
            )

        # -------------------------
        # hybrid routing
        # -------------------------

        candidates = self.route_retrieval(query_type, candidates)

        # -------------------------
        # temporal expansion
        # -------------------------

        if self.is_temporal_query(final_query):
            candidates.extend(self.temporal_candidates(candidates))

        # -------------------------
        # deduplicate candidates
        # -------------------------

        seen = set()
        unique = []

        for d in candidates:

            i = id(d)

            if i not in seen:
                unique.append(d)
                seen.add(i)

        candidates = unique

        # -------------------------
        # stage 1 ranking (cheap)
        # -------------------------

        stage1 = []

        threshold = 0.03 if len(candidates) < 200 else 0.05

        for d in candidates:

            vec_score = cosine_similarity(q_vec, d["vector"])

            if vec_score < threshold:
                continue

            tokens = d["tokens"]

            kw_score = keyword_score(query_tokens, tokens)

            quick_score = 0.7 * vec_score + 0.3 * kw_score

            stage1.append((quick_score, vec_score, d))

        stage1.sort(key=lambda x: x[0], reverse=True)

        limit = max(40, k * 10)

        stage1 = stage1[:limit]

        # -------------------------
        # stage 2 ranking (expensive)
        # -------------------------

        heap = []

        for _, vec_score, d in stage1:

            doc_tokens = d["tokens"]

            kw_score = keyword_score(query_tokens, doc_tokens)
            ent_score = entity_boost(query, d["text"])
            ctx_score = contextual_score(doc_tokens)

            narrative_score = self.narrative_priority_score(
                doc_tokens,
                event_tokens
            )

            temporal = self.temporal_score(d)

            final_score = (
                0.45 * vec_score
                + 0.20 * kw_score
                + 0.10 * ent_score
                + 0.10 * ctx_score
                + 0.10 * narrative_score
                + 0.05 * temporal
            )

            if len(heap) < k:
                heapq.heappush(heap, (final_score, d))
            else:
                heapq.heappushpop(heap, (final_score, d))

        heap.sort(reverse=True)

        result = [d["text"] for _, d in heap]

        self.semantic_cache.set(final_query, q_vec, result)

        return result