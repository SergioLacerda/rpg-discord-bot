import bisect
from pathlib import Path

from rpgbot.infrastructure.embedding_cache import embed
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
from rpgbot.utils import load_json


VECTOR_FILE = Path("campaign/index_vectors.json")
EVENT_FILE = Path("campaign/memory/events.json")


class VectorIndex:

    def __init__(self, vector_file=VECTOR_FILE):

        self.vector_file = vector_file

        self.docs = []
        self.lsh_buckets = {}
        self.projections = []

    # ---------------------------------------------------------
    # Load index
    # ---------------------------------------------------------

    def load(self):

        docs = load_json(self.vector_file, [])

        docs.sort(key=lambda d: d["proj"])

        for d in docs:

            if "tokens" not in d:
                d["tokens"] = tokenize(d["text"])

            if "token_set" not in d:
                d["token_set"] = set(d["tokens"])

        self.docs = docs
        self.projections = [d["proj"] for d in docs]

        buckets = {}

        for d in docs:

            key = d.get("lsh")

            if key:
                buckets.setdefault(key, []).append(d)

        self.lsh_buckets = buckets

    # ---------------------------------------------------------
    # Narrative priority
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

        if not event_tokens:
            return 0.0

        overlap = set(doc_tokens) & event_tokens

        if not overlap:
            return 0.0

        return min(1.0, len(overlap) * 0.15)

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    async def search(self, query, q_vec=None, k=4):

        if len(query) < 4:
            return []

        if not self.docs:
            self.load()

        if q_vec is None:
            expanded_query = expand_query(query)
            q_vec = await embed(expanded_query)

        bucket = lsh_hash(q_vec)

        candidates = self.lsh_buckets.get(bucket, [])

        # ---------------------------------------------------------
        # fallback ANN + entidade
        # ---------------------------------------------------------

        if not candidates:

            pos = bisect.bisect_left(self.projections, project(q_vec))

            window = 50

            start = max(0, pos - window)
            end = min(len(self.docs), pos + window)

            candidates = self.docs[start:end]

            related = {e.lower() for e in related_entities(query)}

            if related:

                for d in self.docs[:200]:

                    if d["token_set"] & related:
                        candidates.append(d)

        # limitar candidatos
        if len(candidates) > 200:
            candidates = candidates[:200]

        # ---------------------------------------------------------
        # VECTOR PREFILTER (nova melhoria)
        # ---------------------------------------------------------

        prefiltered = []

        for d in candidates:

            vec_score = cosine_similarity(q_vec, d["vector"])

            if vec_score > 0.20:  # limiar barato
                prefiltered.append((vec_score, d))

        if prefiltered:
            prefiltered.sort(key=lambda x: x[0], reverse=True)
            candidates = [d for _, d in prefiltered[:80]]

        # ---------------------------------------------------------
        # Ranking híbrido
        # ---------------------------------------------------------

        query_tokens = tokenize(query)
        event_tokens = self.load_recent_event_tokens()

        scored = []

        EARLY_EXIT_THRESHOLD = 0.95

        for d in candidates:

            vec_score = cosine_similarity(q_vec, d["vector"])

            doc_tokens = d.get("tokens")

            if doc_tokens is None:
                doc_tokens = tokenize(d["text"])

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

            scored.append((final_score, d))

            # ---------------------------------------------------------
            # EARLY EXIT
            # ---------------------------------------------------------

            if len(scored) >= k:

                top = sorted(scored, key=lambda x: x[0], reverse=True)[:k]

                if top[-1][0] >= EARLY_EXIT_THRESHOLD:
                    break

        scored.sort(key=lambda x: x[0], reverse=True)

        return [d["text"] for _, d in scored[:k]]