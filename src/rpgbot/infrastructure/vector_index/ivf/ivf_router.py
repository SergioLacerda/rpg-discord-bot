import heapq

from rpgbot.utils.vector.vector_utils import cosine_similarity


class IVFRouter:
    """
    Router IVF otimizado.

    Melhorias:
    - suporta subset (entity prefilter)
    - evita sort completo de centróides
    - reduz lookups repetidos
    - early stopping mais eficiente
    """

    def __init__(
        self,
        ivf_index,
        vector_store,
        n_probe=6,
        max_candidates=800
    ):

        self.index = ivf_index
        self.vector_store = vector_store

        self.n_probe = n_probe
        self.max_candidates = max_candidates


    def search(self, q_vec, subset=None):

        centroids = self.index.centroids
        inverted_lists = self.index.inverted_lists
        doc_to_cluster = self.index.doc_to_cluster

        # -------------------------------------------------
        # calcular similaridade com centróides
        # -------------------------------------------------

        centroid_scores = []

        for cid, centroid in enumerate(centroids):

            s = cosine_similarity(q_vec, centroid)

            centroid_scores.append((s, cid))

        # pegar apenas os melhores clusters
        probe_clusters = [
            cid for _, cid in heapq.nlargest(self.n_probe, centroid_scores)
        ]

        # -------------------------------------------------
        # coletar candidatos
        # -------------------------------------------------

        candidates = []

        subset_set = set(subset) if subset else None

        for cid in probe_clusters:

            doc_ids = inverted_lists.get(cid, [])

            if subset_set:
                for doc_id in doc_ids:

                    if doc_id in subset_set:
                        candidates.append(doc_id)

            else:
                candidates.extend(doc_ids)

            if len(candidates) >= self.max_candidates:
                break

        if not candidates:
            return []

        # -------------------------------------------------
        # ordenar candidatos por proximidade do cluster
        # -------------------------------------------------

        cluster_priority = {
            cid: rank
            for rank, (_, cid) in enumerate(
                heapq.nlargest(self.n_probe, centroid_scores)
            )
        }

        candidates.sort(
            key=lambda doc_id: cluster_priority.get(
                doc_to_cluster.get(doc_id, -1),
                9999
            )
        )

        return candidates[: self.max_candidates]