class TemporalPriorityStage:
    """
    Prioriza eventos recentes quando a query é temporal.
    Reduz o número de candidatos enviados para o ranking.
    """

    def __init__(self, max_candidates=120):

        self.max_candidates = max_candidates

    # ---------------------------------------------------------
    # temporal detection
    # ---------------------------------------------------------

    def is_temporal_query(self, query: str):

        q = query.lower()

        patterns = (
            "quando",
            "quando aconteceu",
            "recentemente",
            "último",
            "ultimo",
            "recent",
            "when",
            "timeline",
        )

        return any(p in q for p in patterns)

    # ---------------------------------------------------------
    # pipeline stage
    # ---------------------------------------------------------

    def run(self, ctx, candidates):

        if not candidates:
            return candidates

        if not self.is_temporal_query(ctx.query):
            return candidates

        metadata = ctx.metadata_store

        if not metadata:
            return candidates

        # ordenar por recência
        candidates = sorted(
            candidates,
            key=lambda doc_id: ctx.temporal_index.recency_score(doc_id),
            reverse=True
        )

        # reduzir candidatos
        return candidates[: self.max_candidates]