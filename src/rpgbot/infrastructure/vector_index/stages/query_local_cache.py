from collections import OrderedDict


class QueryLocalCache:
    """
    Cache local de candidatos baseado em similaridade de queries.
    Reduz chamadas repetidas ao retrieval.
    """

    def __init__(self, size=32, similarity_threshold=0.92):

        self.size = size
        self.threshold = similarity_threshold

        self.cache = OrderedDict()

    # ---------------------------------------------------------
    # cosine similarity
    # ---------------------------------------------------------

    def similarity(self, a, b):

        return sum(x*y for x, y in zip(a, b))

    # ---------------------------------------------------------
    # pipeline stage
    # ---------------------------------------------------------

    def run(self, ctx, candidates):

        q_vec = ctx.q_vec

        # procurar query similar no cache
        for vec, cached_candidates in self.cache.values():

            if self.similarity(q_vec, vec) > self.threshold:
                return cached_candidates.copy()

        # se não encontrou, deixar pipeline continuar
        return candidates

    # ---------------------------------------------------------
    # update cache
    # ---------------------------------------------------------

    def update(self, ctx, candidates):

        key = hash(tuple(ctx.q_vec[:8]))

        self.cache[key] = (ctx.q_vec, candidates)

        if len(self.cache) > self.size:
            self.cache.popitem(last=False)