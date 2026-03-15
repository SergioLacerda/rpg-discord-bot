class ClusterManager:
    """
    Gerencia clusters e garante consistência com VectorStore.
    """

    def __init__(self, cluster_builder, drift_detector):

        self.cluster_builder = cluster_builder
        self.drift_detector = drift_detector

        self.centroids = []
        self.centroid_projections = []
        self.cluster_docs = {}
        self.cluster_indexes = {}

        self.last_cluster_size = 0


    def build_hnsw_clusters(self, vector_store):

        from rpgbot.infrastructure.vector_index.retrieval.hnsw_cluster_index import (
            HNSWClusterIndex
        )

        self.cluster_indexes = {}

        for cid, docs in self.clusters.items():

            index = HNSWClusterIndex(dim=self.vector_dim)

            for doc_id in docs:

                vec = vector_store.get(doc_id)

                if vec is not None:
                    index.add(doc_id, vec)

            self.cluster_indexes[cid] = index


    def rebuild(self, doc_ids, vector_store):

        vectors = [
            vector_store.get(doc_id)
            for doc_id in doc_ids
        ]

        clusters = self.cluster_builder.build(
            doc_ids,
            vectors
        )

        self.centroids = clusters.centroids
        self.centroid_projections = clusters.centroid_projections
        self.cluster_docs = clusters.cluster_docs

        self.last_cluster_size = len(doc_ids)

        return clusters


    def update(self, doc_ids, vector_store):

        rebuild = self.drift_detector.should_rebuild(
            current_size=len(doc_ids),
            previous_size=self.last_cluster_size,
            has_centroids=bool(self.centroids),
        )

        if rebuild:

            self.rebuild(doc_ids, vector_store)

            # -----------------------------------------
            # build HNSW per cluster
            # -----------------------------------------

            if hasattr(self, "build_hnsw_clusters"):
                self.build_hnsw_clusters(vector_store)

            self.last_cluster_size = len(doc_ids)

            return True

        return False