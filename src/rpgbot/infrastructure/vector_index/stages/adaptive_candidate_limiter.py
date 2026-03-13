
class AdaptiveCandidateLimiter:
    """
    Limita dinamicamente o número de candidatos
    baseado no tipo de consulta.
    """

    def __init__(
        self,
        semantic_limit=150,
        memory_limit=80,
        lore_limit=120,
        investigation_limit=250,
    ):

        self.semantic_limit = semantic_limit
        self.memory_limit = memory_limit
        self.lore_limit = lore_limit
        self.investigation_limit = investigation_limit

    # ---------------------------------------------------------
    # pipeline stage
    # ---------------------------------------------------------

    def run(self, ctx, candidates):

        if not candidates:
            return candidates

        qt = ctx.query_type

        # ordenar por recência se for query temporal
        if qt == "memory" and ctx.temporal_index and len(candidates) > self.memory_limit:

            candidates = sorted(
                candidates,
                key=lambda doc_id: ctx.temporal_index.recency_score(doc_id),
                reverse=True
            )

        if qt == "memory":
            limit = self.memory_limit

        elif qt == "lore":
            limit = self.lore_limit

        elif qt == "investigation":
            limit = self.investigation_limit

        else:
            limit = self.semantic_limit

        if len(candidates) <= limit:
            return candidates

        return candidates[:limit]