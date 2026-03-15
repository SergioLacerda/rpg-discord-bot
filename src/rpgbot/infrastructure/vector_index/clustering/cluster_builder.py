import random
from dataclasses import dataclass

from rpgbot.utils.vector.vector_math import cosine_similarity
from rpgbot.utils.vector.vector_utils import project


@dataclass
class ClusterResult:
    centroids: list
    centroid_projections: list
    cluster_docs: dict
    cluster_count: int
    super_cluster_count: int
    size: int


class ClusterBuilder:

    def __init__(
        self,
        min_clusters=8,
        max_clusters=512,
        min_super_clusters=4,
        max_super_clusters=64,
    ):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_super_clusters = min_super_clusters
        self.max_super_clusters = max_super_clusters

    # ---------------------------------------------------------
    # adaptive cluster sizing
    # ---------------------------------------------------------

    def adaptive_cluster_sizes(self, n):

        if n < 50:
            return 4, 2

        cluster_target = int(n ** 0.5)
        cluster_target = max(self.min_clusters, min(cluster_target, self.max_clusters))

        super_target = int(cluster_target ** 0.5)
        super_target = max(self.min_super_clusters, min(super_target, self.max_super_clusters))

        return cluster_target, super_target

    # ---------------------------------------------------------
    # build clusters
    # ---------------------------------------------------------

    def build(self, docs):

        if not docs:
            return ClusterResult([], [], {}, 0, 0, 0)

        n = len(docs)

        cluster_count, super_cluster_count = self.adaptive_cluster_sizes(n)

        vectors = [d["vector"] for d in docs]

        centroids = random.sample(
            vectors,
            min(cluster_count, len(vectors))
        )

        # kmeans-lite refinement

        for _ in range(2):

            cluster_docs = {i: [] for i in range(len(centroids))}

            for doc in docs:

                best = 0
                best_score = -1

                for i, c in enumerate(centroids):

                    s = cosine_similarity(doc["vector"], c)

                    if s > best_score:
                        best_score = s
                        best = i

                cluster_docs[best].append(doc)

            new_centroids = []

            for i in range(len(centroids)):

                bucket = cluster_docs[i]

                if not bucket:
                    new_centroids.append(centroids[i])
                    continue

                dim = len(bucket[0]["vector"])
                mean = [0.0] * dim

                for d in bucket:
                    vec = d["vector"]
                    for j in range(dim):
                        mean[j] += vec[j]

                mean = [v / len(bucket) for v in mean]

                new_centroids.append(mean)

            centroids = new_centroids

        # projection routing

        centroid_projections = [
            project(c) for c in centroids
        ]

        pairs = sorted(
            zip(centroid_projections, centroids),
            key=lambda x: x[0]
        )

        centroid_projections = [p for p, _ in pairs]
        centroids = [c for _, c in pairs]

        return ClusterResult(
            centroids=centroids,
            centroid_projections=centroid_projections,
            cluster_docs=cluster_docs,
            cluster_count=cluster_count,
            super_cluster_count=super_cluster_count,
            size=n
        )