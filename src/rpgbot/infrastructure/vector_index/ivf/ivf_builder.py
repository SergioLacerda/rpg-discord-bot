import random

from rpgbot.utils.vector.vector_math import cosine_similarity


class IVFBuilder:

    def __init__(
        self,
        n_clusters=256,
        max_train_samples=20000,
        iterations=10
    ):

        self.n_clusters = n_clusters
        self.max_train_samples = max_train_samples
        self.iterations = iterations


    def build(self, doc_ids, vector_store):

        # -------------------------------------------------
        # carregar vetores
        # -------------------------------------------------

        vectors = [
            vector_store.get(doc_id)
            for doc_id in doc_ids
        ]

        # -------------------------------------------------
        # sample training
        # -------------------------------------------------

        if len(vectors) > self.max_train_samples:

            sample = random.sample(
                vectors,
                self.max_train_samples
            )

        else:
            sample = vectors

        # -------------------------------------------------
        # treinar centróides
        # -------------------------------------------------

        centroids = self._kmeans(sample)

        # -------------------------------------------------
        # atribuir documentos aos clusters
        # -------------------------------------------------

        inverted_lists = {i: [] for i in range(len(centroids))}
        doc_to_cluster = {}

        for doc_id in doc_ids:

            vec = vector_store.get(doc_id)

            best = -1
            best_score = -1

            for cid, centroid in enumerate(centroids):

                s = cosine_similarity(vec, centroid)

                if s > best_score:
                    best_score = s
                    best = cid

            inverted_lists[best].append(doc_id)
            doc_to_cluster[doc_id] = best

        return IVFIndex(
            centroids=centroids,
            inverted_lists=inverted_lists,
            doc_to_cluster=doc_to_cluster
        )


    def _kmeans(self, vectors):

        # inicialização simples
        centroids = random.sample(vectors, self.n_clusters)

        for _ in range(self.iterations):

            clusters = [[] for _ in centroids]

            for v in vectors:

                best = 0
                best_score = -1

                for cid, c in enumerate(centroids):

                    s = cosine_similarity(v, c)

                    if s > best_score:
                        best_score = s
                        best = cid

                clusters[best].append(v)

            new_centroids = []

            for cluster in clusters:

                if not cluster:
                    new_centroids.append(random.choice(vectors))
                    continue

                dim = len(cluster[0])

                mean = [0.0] * dim

                for v in cluster:
                    for i in range(dim):
                        mean[i] += v[i]

                size = len(cluster)

                mean = [x / size for x in mean]

                new_centroids.append(mean)

            centroids = new_centroids

        return centroids