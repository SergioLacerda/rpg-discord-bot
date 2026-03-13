import heapq


class LazyVectorSimilarity:
    """
    Calcula similaridade vetorial apenas quando necessário.
    Evita computar similarity para todos os candidatos.
    """

    def __init__(self, vector_store):

        self.vector_store = vector_store

    def similarity(self, q_vec, doc_id):

        vec = self.vector_store.get(doc_id)

        if vec is None:
            return 0.0

        return sum(a * b for a, b in zip(q_vec, vec))

    def top_k(self, q_vec, candidate_ids, k=50):

        heap = []

        for doc_id in candidate_ids:

            score = self.similarity(q_vec, doc_id)

            if len(heap) < k:
                heapq.heappush(heap, (score, doc_id))
            else:
                heapq.heappushpop(heap, (score, doc_id))

        heap.sort(reverse=True)

        return [doc_id for _, doc_id in heap]