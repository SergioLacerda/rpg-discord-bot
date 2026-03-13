class HierarchicalCandidateReducer:
    """
    Reduz candidatos usando múltiplos sinais
    antes do ranking pesado.
    """

    def __init__(self, max_candidates=120):
        self.max_candidates = max_candidates

    def run(self, ctx, candidates):

        if not candidates:
            return candidates

        if len(candidates) <= self.max_candidates:
            return candidates

        vector_store = ctx.vector_store
        token_store = ctx.token_store
        metadata = ctx.metadata_store
        temporal = getattr(ctx, "temporal_index", None)
        ivf_router = getattr(ctx, "ivf_router", None)

        query_tokens = set(ctx.query_tokens)

        scored = []

        for doc_id in candidates:

            score = 0.0

            # lexical overlap
            tokens = token_store.get(doc_id)
            if tokens:
                overlap = len(query_tokens & set(tokens))
                score += 0.3 * overlap

            # recency
            if temporal:
                score += 0.3 * temporal.recency_score(doc_id)

            # cluster proximity
            if ivf_router:
                score += 0.4 * ivf_router.cluster_similarity(ctx.q_vec, doc_id)

            scored.append((score, doc_id))

        scored.sort(reverse=True)

        return [doc_id for _, doc_id in scored[: self.max_candidates]]