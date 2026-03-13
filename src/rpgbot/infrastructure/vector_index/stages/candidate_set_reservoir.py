import heapq


class CandidateSetReservoir:

    def __init__(self, max_size=300):
        self.max_size = max_size

    def run(self, ctx, candidates):

        if not candidates:
            return candidates

        if len(candidates) <= self.max_size * 2:
            return candidates

        vector_store = ctx.vector_store
        q_vec = ctx.q_vec

        ivf_router = getattr(ctx, "ivf_router", None)

        heap = []

        for doc_id in candidates:

            vec = vector_store.get(doc_id)
            if vec is None:
                continue

            # base similarity
            score = sum(a * b for a, b in zip(q_vec, vec))

            # IVF bonus
            if ivf_router:
                cluster_score = ivf_router.cluster_similarity(q_vec, doc_id)
                score += 0.15 * cluster_score

            if len(heap) < self.max_size:
                heapq.heappush(heap, (score, doc_id))
            else:
                heapq.heappushpop(heap, (score, doc_id))

        return [doc_id for _, doc_id in heap]