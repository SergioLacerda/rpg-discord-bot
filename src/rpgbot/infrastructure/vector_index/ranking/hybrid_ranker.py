class HybridRanker:
    """
    Combina múltiplos rankings usando Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, k=60):
        self.k = k


    def reciprocal_rank_fusion(self, *rank_lists, limit=None):
        """
        rank_lists: listas ordenadas de documentos
        limit: número máximo de resultados
        """

        scores = {}

        for rank_list in rank_lists:

            if not rank_list:
                continue

            for rank, doc in enumerate(rank_list):

                doc_id = doc["id"]

                score = 1 / (self.k + rank)

                scores[doc_id] = scores.get(doc_id, 0) + score

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        seen = set()

        for doc_id, _ in ranked:

            for rank_list in rank_lists:

                for d in rank_list:

                    if d["id"] == doc_id and doc_id not in seen:
                        results.append(d)
                        seen.add(doc_id)
                        break

                if doc_id in seen:
                    break

            if limit and len(results) >= limit:
                break

        return results


    def fuse(self, vector_candidates, lexical_candidates=None, graph_candidates=None, limit=None):

        rank_lists = [vector_candidates]

        if lexical_candidates:
            rank_lists.append(lexical_candidates)

        if graph_candidates:
            rank_lists.append(graph_candidates)

        return self.reciprocal_rank_fusion(
            *rank_lists,
            limit=limit
        )