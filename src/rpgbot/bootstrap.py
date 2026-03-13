from rpgbot.core.container import container
from rpgbot.core.paths import CAMPAIGN_DIR

from rpgbot.rag.semantic_cache import HierarchicalSemanticCache
from rpgbot.rag.retrieval_engine import RetrievalEngine

from rpgbot.infrastructure.embedding_cache import embed
from rpgbot.utils.hash_utils import sha256_hash

from rpgbot.infrastructure.vector_index.factory import build_vector_index
from rpgbot.infrastructure.vector_index.components import VectorIndexComponents

from rpgbot.infrastructure.vector_index.storage.document_loader import DocumentLoader
from rpgbot.infrastructure.vector_index.storage.document_repository import DocumentRepository
from rpgbot.infrastructure.vector_index.storage.feature_store import FeatureStore

from rpgbot.infrastructure.vector_index.indexing.embedding_indexer import EmbeddingIndexer

from rpgbot.infrastructure.vector_index.stores.vector_store_mmap import MemoryMappedVectorStore
from rpgbot.infrastructure.vector_index.stores.document_store import DocumentStore
from rpgbot.infrastructure.vector_index.stores.token_store import TokenStore
from rpgbot.infrastructure.vector_index.stores.metadata_store import MetadataStore

from rpgbot.infrastructure.vector_index.retrieval.query_classifier import QueryClassifier

from rpgbot.infrastructure.vector_index.ranking.stage1_ranker import Stage1Ranker
from rpgbot.infrastructure.vector_index.ranking.stage2_ranker import Stage2Ranker

from rpgbot.infrastructure.vector_index.clustering.cluster_manager import ClusterManager
from rpgbot.infrastructure.vector_index.clustering.cluster_builder import ClusterBuilder
from rpgbot.infrastructure.vector_index.clustering.drift_detection import ClusterDriftDetector

from rpgbot.infrastructure.vector_index.ivf.ivf_builder import IVFBuilder
from rpgbot.infrastructure.vector_index.ivf.ivf_router import IVFRouter


_bootstrapped = False


# ---------------------------------------------------------
# container setup
# ---------------------------------------------------------

def setup_container():
    if container._providers:
        return

    # --------------------------------------------------
    # core
    # --------------------------------------------------

    container.register("semantic_cache", HierarchicalSemanticCache)

    container.register(
        "embed",
        lambda: embed,
        singleton=True
    )
    container.register("hash", lambda: sha256_hash)

    # --------------------------------------------------
    # stores
    # --------------------------------------------------

    container.register("vector_store", MemoryMappedVectorStore)
    container.register("document_store", DocumentStore)
    container.register("token_store", TokenStore)
    container.register("metadata_store", MetadataStore)
    container.register("feature_store", lambda: FeatureStore(CAMPAIGN_DIR))

    # --------------------------------------------------
    # indexing
    # --------------------------------------------------

    container.register("document_loader", lambda: DocumentLoader(CAMPAIGN_DIR))

    container.register(
        "repository",
        lambda: DocumentRepository(CAMPAIGN_DIR / "vector.index")
    )

    container.register("embedding_indexer", EmbeddingIndexer)

    # --------------------------------------------------
    # query
    # --------------------------------------------------

    container.register(
        "query_classifier",
        QueryClassifier,
        singleton=True
    )

    # --------------------------------------------------
    # ranking
    # --------------------------------------------------

    container.register("stage1_ranker", Stage1Ranker)
    container.register(
        "stage2_ranker",
        lambda feature_store: Stage2Ranker(feature_store)
    )

    # --------------------------------------------------
    # clustering
    # --------------------------------------------------

    container.register("cluster_builder", ClusterBuilder)
    container.register("drift_detector", ClusterDriftDetector)
    container.register("cluster_manager", ClusterManager)

    # --------------------------------------------------
    # ANN
    # --------------------------------------------------

    container.register("ivf_builder", IVFBuilder)

    container.register(
        "ivf_router",
        lambda vector_store: IVFRouter(None, vector_store)
    )

    # --------------------------------------------------
    # VectorIndex
    # --------------------------------------------------

    container.register("retrieval_engine", lambda: RetrievalEngine(container.resolve("vector_index")))


    container.register(
        "vector_index_factory",
        lambda campaign_id: build_vector_index_service(campaign_id)
    )

    container.register("vector_index", build_vector_index_service())

    _bootstrapped = True


# ---------------------------------------------------------
# VectorIndex factory
# ---------------------------------------------------------

@container.inject
def build_vector_index_service(
    query_classifier,
    stage1_ranker,
    stage2_ranker,
    cluster_manager,
    vector_store,
    document_store,
    token_store,
    metadata_store,
    ivf_builder,
    ivf_router,
    semantic_cache,
    document_loader,
    repository,
    embedding_indexer,
    feature_store
):

    components = VectorIndexComponents(
        query_classifier=query_classifier,
        stage1_ranker=stage1_ranker,
        stage2_ranker=stage2_ranker,
        cluster_manager=cluster_manager,
        vector_store=vector_store,
        document_store=document_store,
        token_store=token_store,
        metadata_store=metadata_store,
        ivf_builder=ivf_builder,
        ivf_router=ivf_router,
    )

    return build_vector_index(
        components=components,
        semantic_cache=semantic_cache,
        document_loader=document_loader,
        repository=repository,
        embedding_indexer=embedding_indexer,
        feature_store=feature_store,
        campaign_dir=CAMPAIGN_DIR,
    )


# ---------------------------------------------------------
# RetrievalEngine factory
# ---------------------------------------------------------

def build_retrieval_engine():

    setup_container()

    index = container.resolve("vector_index")

    engine = RetrievalEngine(index)

    container.register("retrieval_engine", lambda: engine)

    return engine