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

        self.last_cluster_size = 0


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

        if self.drift_detector.should_rebuild(
            current_size=len(doc_ids),
            previous_size=self.last_cluster_size,
            has_centroids=bool(self.centroids),
        ):
            return self.rebuild(doc_ids, vector_store)

        return None