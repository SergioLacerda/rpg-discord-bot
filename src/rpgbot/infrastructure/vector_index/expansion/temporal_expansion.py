class TemporalExpansion:
    """
    Expande candidatos quando a consulta possui intenção temporal
    (ex: 'o que aconteceu recentemente', 'quando ocorreu', etc.).
    """

    def __init__(self, max_expansion=20, time_window=None):
        self.max_expansion = max_expansion
        self.time_window = time_window

    # ---------------------------------------------------------
    # temporal query detection
    # ---------------------------------------------------------

    def is_temporal_query(self, query: str) -> bool:

        q = query.lower()

        patterns = (
            "quando",
            "quando aconteceu",
            "recentemente",
            "último",
            "ultimo",
            "recent",
            "when",
            "timeline"
        )

        return any(p in q for p in patterns)

    # ---------------------------------------------------------
    # pipeline stage
    # ---------------------------------------------------------

    def run(self, ctx, candidates):

        if ctx.query_type != "memory":
            return candidates

        if not self.is_temporal_query(ctx.query):
            return candidates

        metadata_store = ctx.metadata_store

        if not metadata_store:
            return candidates

        if not candidates:
            return candidates

        # evitar crescimento explosivo
        if len(candidates) > 200:
            return candidates

        expanded = []
        seen = set(candidates)

        # ordenar candidatos por timestamp
        sorted_docs = sorted(
            candidates,
            key=lambda doc_id: metadata_store.get_timestamp(doc_id)
        )

        for doc_id in sorted_docs:

            ts = metadata_store.get_timestamp(doc_id)

            if ts is None:
                continue

            for other in sorted_docs:

                if other == doc_id:
                    continue

                if other in seen:
                    continue

                other_ts = metadata_store.get_timestamp(other)

                if other_ts is None:
                    continue

                if self.time_window:

                    if abs(other_ts - ts) > self.time_window:
                        continue

                expanded.append(other)
                seen.add(other)

                if len(expanded) >= self.max_expansion:
                    return candidates + expanded