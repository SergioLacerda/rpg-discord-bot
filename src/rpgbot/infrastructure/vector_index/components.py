from dataclasses import dataclass


@dataclass
class VectorIndexComponents:

    def __init__(
        self,
        query_classifier,
        stage1_ranker,
        stage2_ranker,
        cluster_manager,
        vector_store,
        document_store,
        token_store,
        metadata_store,
        ivf_builder,
        ivf_router
    ):

        self.query_classifier = query_classifier
        self.stage1_ranker = stage1_ranker
        self.stage2_ranker = stage2_ranker
        self.cluster_manager = cluster_manager

        self.vector_store = vector_store
        self.document_store = document_store
        self.token_store = token_store
        self.metadata_store = metadata_store

        self.ivf_builder = ivf_builder
        self.ivf_router = ivf_router