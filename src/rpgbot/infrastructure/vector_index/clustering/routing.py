import bisect

from rpgbot.utils.vector.vector_utils import cosine_similarity, project


class ClusterRouter:

    def __init__(self, centroids, centroid_projections, cluster_docs):
        self.centroids = centroids
        self.centroid_projections = centroid_projections
        self.cluster_docs = cluster_docs


    def projection_routing(self, q_vec, top_k=20):

        if not self.centroid_projections:
            return []

        q_proj = project(q_vec)

        pos = bisect.bisect_left(self.centroid_projections, q_proj)

        window = max(5, top_k)

        start = max(0, pos - window)
        end = min(len(self.centroid_projections), pos + window)

        return list(range(start, end))


    def cluster_candidates(self, q_vec, top_clusters=3):

        candidate_ids = self.projection_routing(q_vec)

        scored = []

        for i in candidate_ids:

            centroid = self.centroids[i]

            s = cosine_similarity(q_vec, centroid)

            scored.append((s, i))

        scored.sort(reverse=True)

        docs = []

        for _, cid in scored[:top_clusters]:
            docs.extend(self.cluster_docs.get(cid, []))

        return docs