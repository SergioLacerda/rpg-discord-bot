import heapq

from rpgbot.infrastructure.vector_index.storage.feature_store import FeatureStore
from rpgbot.infrastructure.vector_index.stores.temporal_memory_index import TemporalMemoryIndex
from rpgbot.utils.vector.vector_utils import keyword_score


class Stage2Ranker:

    def __init__(
        self,
        feature_store: FeatureStore,
        gate_ratio=0.55,
        expensive_limit=40
    ):

        self.feature_store = feature_store
        self.gate_ratio = gate_ratio
        self.expensive_limit = expensive_limit

    # ---------------------------------------------------------
    # ranking
    # ---------------------------------------------------------

    def rank(
        self,
        stage1,
        query_tokens,
        query_entities,
        event_tokens,
        vector_index,
        k=4
    ):

        heap = []

        quick_scores = []

        feature_get = self.feature_store.get
        kw_score_fn = keyword_score
        temporal_idx = vector_index.temporal_index

        best_score = None

        # ---------------------------------------------------------
        # first pass (cheap features)
        # ---------------------------------------------------------

        for _, vec_score, d in stage1:

            doc_id = d["id"]
            doc_tokens = d["tokens"]

            features = feature_get(doc_id)

            # cheap features
            ent_score = len(query_entities & features["entities"])
            kw_score = kw_score_fn(query_tokens, doc_tokens)

            quick_score = (
                0.45 * vec_score
                + 0.20 * kw_score
                + 0.10 * ent_score
            )

            quick_scores.append((quick_score, vec_score, d))

            if best_score is None or quick_score > best_score:
                best_score = quick_score

        quick_scores.sort(reverse=True)

        # ---------------------------------------------------------
        # feature gate
        # ---------------------------------------------------------

        gated = []

        threshold = best_score * self.gate_ratio if best_score else 0

        for quick_score, vec_score, d in quick_scores:

            if quick_score < threshold:
                continue

            gated.append((quick_score, vec_score, d))

            if len(gated) >= self.expensive_limit:
                break

        # ---------------------------------------------------------
        # second pass (expensive features)
        # ---------------------------------------------------------

        for quick_score, vec_score, d in gated:

            doc_id = d["id"]
            doc_tokens = d["tokens"]

            features = feature_get(doc_id)

            ctx_score = features["context"]

            narrative_score = vector_index.narrative_priority_score(
                doc_tokens,
                event_tokens
            )

            temporal = 0.0
            if temporal_idx:
                temporal = temporal_idx.recency_score(doc_id)

            final_score = (
                quick_score
                + 0.10 * ctx_score
                + 0.10 * narrative_score
                + 0.05 * temporal
            )

            if len(heap) < k:
                heapq.heappush(heap, (final_score, d))
            else:
                heapq.heappushpop(heap, (final_score, d))

        heap.sort(reverse=True)

        return [d for _, d in heap]