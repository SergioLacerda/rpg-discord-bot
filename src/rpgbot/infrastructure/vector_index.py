import bisect
import hashlib
import heapq
import random
import re
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


'''
query
↓
semantic cache
↓
query expansion
↓
cluster coarse search
↓
HNSW navigation
↓
LSH bucket
↓
ANN prefilter
↓
vector prefilter
↓
lexical retrieval
↓
graph traversal
↓
RRF fusion
↓
ranking final
'''

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
        self.projections = []
        self.lsh_buckets = {}

        self.inverted_index = {}

        self.centroids = []
        self.cluster_docs = {}
        self.cluster_count = 16

        self.hnsw_index = None
        self.ann_index = None

        self.entity_graph = {}
        self.doc_entities = {}

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()

    # ---------------------------------------------------------
    # entity graph
    # ---------------------------------------------------------

    def _build_entity_graph(self):

        graph = defaultdict(set)
        doc_entities = {}

        ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")

        for doc_id, d in enumerate(self.docs):

            entities = {
                e.lower()
                for e in ENTITY_RE.findall(d["text"])
            }

            doc_entities[doc_id] = entities

            for e in entities:
                graph[e].add(doc_id)

        entity_graph = defaultdict(set)

        for doc_id, ents in doc_entities.items():
            for e in ents:
                for r in graph[e]:
                    if r != doc_id:
                        entity_graph[doc_id].add(r)

        self.entity_graph = entity_graph
        self.doc_entities = doc_entities

    # ---------------------------------------------------------
    # clustering (coarse stage)
    # ---------------------------------------------------------

    def _build_clusters(self):

        if not self.docs:
            return

        vectors = [d["vector"] for d in self.docs]

        self.centroids = random.sample(
            vectors,
            min(self.cluster_count, len(vectors))
        )

        self.cluster_docs = {i: [] for i in range(len(self.centroids))}

        for doc in self.docs:

            best = 0
            best_score = -1

            for i, c in enumerate(self.centroids):

                s = cosine_similarity(doc["vector"], c)

                if s > best_score:
                    best_score = s
                    best = i

            self.cluster_docs[best].append(doc)

    def coarse_candidates(self, q_vec, clusters=3):

        if not self.centroids:
            return []

        scores = []

        for i, c in enumerate(self.centroids):

            s = cosine_similarity(q_vec, c)
            scores.append((s, i))

        scores.sort(reverse=True)

        selected = []

        for _, idx in scores[:clusters]:
            selected.extend(self.cluster_docs.get(idx, []))

        return selected

    # ---------------------------------------------------------
    # load / incremental indexing
    # ---------------------------------------------------------

    def load(self):

        campaign_path = self.campaign_dir / self.campaign_id

        if not campaign_path.exists():
            self.docs = []
            return
            
        persisted = load_json(self.vector_file, [])

        persisted = [
            d for d in persisted
            if d.get("source", "").startswith(str(self.campaign_dir / self.campaign_id))
        ]

        existing = {
            d["source"]: d
            for d in persisted
            if d.get("source")
        }

        updated_docs = []

        campaign_path = self.campaign_dir / self.campaign_id
        campaign_path.mkdir(parents=True, exist_ok=True)

        for file in campaign_path.glob("**/*.md"):

            source = str(file)

            try:
                text = file.read_text(encoding="utf-8")
                mtime = file.stat().st_mtime
            except OSError:
                continue

            text_hash = self._hash_text(text)

            prev = existing.get(source)

            if prev and prev.get("hash") == text_hash:
                updated_docs.append(prev)
                continue

            vec = self.embed(text)

            new_doc = {
                "text": text,
                "vector": vec,
                "source": source,
                "mtime": mtime,
                "hash": text_hash,
            }

            updated_docs.append(new_doc)

        docs = updated_docs

        inverted = {}

        for doc_id, d in enumerate(docs):

            d["proj"] = project(d["vector"])
            d["lsh"] = lsh_hash(d["vector"])

            if "tokens" not in d:
                d["tokens"] = tokenize(d["text"])

            if "token_set" not in d:
                d["token_set"] = set(d["tokens"])

            for tok in d["tokens"]:
                inverted.setdefault(tok, []).append(doc_id)

        self.inverted_index = inverted

        docs.sort(key=lambda d: d["proj"])

        self.docs = docs
        self.projections = [d["proj"] for d in docs]

        buckets = defaultdict(list)

        for d in docs:
            buckets[d["lsh"]].append(d)

        self.lsh_buckets = buckets

        if docs:
            self.hnsw_index = HNSWIndex(self.docs)
            self.ann_index = self.ann_index_factory(docs)

        self._build_entity_graph()
        self._build_clusters()

        save_json(self.vector_file, docs)

    # ---------------------------------------------------------
    # narrative signals
    # ---------------------------------------------------------

    def load_recent_event_tokens(self, limit=10):

        try:
            events = load_json(EVENT_FILE, [])
        except Exception:
            return set()

        tokens = []

        for e in events[-limit:]:
            tokens.extend(tokenize(e.get("text", "")))

        return set(tokens)

    @staticmethod
    def narrative_priority_score(doc_tokens, event_tokens):

        overlap = set(doc_tokens) & event_tokens

        if not overlap:
            return 0.0

        return min(1.0, len(overlap) * 0.15)

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

        # ---------------------------------------------------------
        # coarse stage
        # ---------------------------------------------------------

        candidates = self.coarse_candidates(q_vec)

        if not candidates:
            candidates = list(self.lsh_buckets.get(lsh_hash(q_vec), []))

        if not candidates and self.hnsw_index:
            candidates = self.hnsw_index.search(q_vec, k=200)

        if not candidates:

            pos = bisect.bisect_left(self.projections, project(q_vec))

            start = max(0, pos - 50)
            end = min(len(self.docs), pos + 50)

            candidates = self.docs[start:end]

        # ---------------------------------------------------------
        # ANN prefilter
        # ---------------------------------------------------------

        if self.ann_index:

            ann_candidates = self.ann_index.search(q_vec)

            if ann_candidates:

                ann_set = {id(d) for d in ann_candidates}
                candidates = [d for d in candidates if id(d) in ann_set]

        # ---------------------------------------------------------
        # vector prefilter
        # ---------------------------------------------------------

        scored = []

        for d in candidates:

            score = cosine_similarity(q_vec, d["vector"])

            if score > 0.05:
                scored.append((score, d))

        scored.sort(reverse=True)

        candidates = [d for _, d in scored[:80]]

        # ---------------------------------------------------------
        # lexical + graph retrieval
        # ---------------------------------------------------------

        lexical = self.lexical_candidates(query_tokens)
        graph_docs = self.graph_candidates(query_tokens)

        if lexical or graph_docs:

            vector_rank = candidates[:100]
            lexical_rank = lexical[:100] if lexical else []
            graph_rank = graph_docs[:100] if graph_docs else []

            merged_vector = vector_rank + graph_rank

            candidates = self.reciprocal_rank_fusion(
                merged_vector,
                lexical_rank,
                limit=150
            )

        # ---------------------------------------------------------
        # ranking final
        # ---------------------------------------------------------

        event_tokens = self.load_recent_event_tokens()

        heap = []

        for d in candidates:

            vec_score = cosine_similarity(q_vec, d["vector"])

            if vec_score < 0.20:
                continue

            doc_tokens = d["tokens"]

            kw_score = keyword_score(query_tokens, doc_tokens)
            ent_score = entity_boost(query, d["text"])
            ctx_score = contextual_score(doc_tokens)

            narrative_score = self.narrative_priority_score(
                doc_tokens,
                event_tokens
            )

            final_score = (
                0.50 * vec_score
                + 0.20 * kw_score
                + 0.10 * ent_score
                + 0.10 * ctx_score
                + 0.10 * narrative_score
            )

            if len(heap) < k:
                heapq.heappush(heap, (final_score, d))
            else:
                heapq.heappushpop(heap, (final_score, d))

        heap.sort(reverse=True)

        result = [d["text"] for _, d in heap]

        self.semantic_cache.set(final_query, q_vec, result)

        return result

    # ---------------------------------------------------------
    # retrieval helpers
    # ---------------------------------------------------------

    def lexical_candidates(self, query_tokens, limit=100):

        scores = {}

        for tok in query_tokens:

            doc_ids = self.inverted_index.get(tok)

            if not doc_ids:
                continue

            for doc_id in doc_ids:
                scores[doc_id] = scores.get(doc_id, 0) + 1

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [self.docs[i] for i, _ in ranked[:limit]]

    def graph_candidates(self, query_tokens, limit=50):

        matched_docs = set()

        for doc_id, ents in self.doc_entities.items():

            if ents.intersection(query_tokens):
                matched_docs.add(doc_id)

        expanded = set(matched_docs)

        for d in matched_docs:
            expanded.update(self.entity_graph.get(d, []))

        return [self.docs[i] for i in list(expanded)[:limit]]

    def reciprocal_rank_fusion(self, vector_rank, lexical_rank, limit=100, k=60):

        scores = {}

        for rank, doc in enumerate(vector_rank):
            scores[id(doc)] = scores.get(id(doc), 0) + 1 / (k + rank + 1)

        for rank, doc in enumerate(lexical_rank):
            scores[id(doc)] = scores.get(id(doc), 0) + 1 / (k + rank + 1)

        merged = {id(doc): doc for doc in vector_rank + lexical_rank}

        ranked = sorted(
            merged.values(),
            key=lambda d: scores[id(d)],
            reverse=True
        )

        return ranked[:limit]