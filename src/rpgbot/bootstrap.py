from rpgbot.core.container import container
from rpgbot.core.config import settings, CAMPAIGN_DIR

from rpgbot.core.providers import llm_registry, embedding_registry
from rpgbot.core.provider_loader import load_providers

from rpgbot.adapters.storage.json_session_repository import AsyncJSONRepository

from rpgbot.rag.semantic_cache import HierarchicalSemanticCache
from rpgbot.rag.retrieval_engine import RetrievalEngine

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

from rpgbot.utils.hash_utils import sha256_hash


_bootstrapped = False


# ---------------------------------------------------------
# providers
# ---------------------------------------------------------

def register_llm():

    def build_llm():

        return llm_registry.create(
            settings.llm.provider,
            api_key=settings.llm.api_key,
            model=settings.llm.model,
            base_url=settings.llm.base_url,
        )

    container.register("llm_provider", build_llm, singleton=True)


def register_embeddings():

    def build_embeddings():

        kwargs = {}

        if settings.embeddings.model:
            kwargs["model"] = settings.embeddings.model

        if settings.embeddings.api_key:
            kwargs["api_key"] = settings.embeddings.api_key

        if settings.embeddings.batch_size:
            kwargs["batch_size"] = settings.embeddings.batch_size

        return embedding_registry.create(
            settings.embeddings.provider,
            **kwargs
        )

    container.register("embedding_provider", build_embeddings, singleton=True)


# ---------------------------------------------------------
# vector index factory
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
# container setup
# ---------------------------------------------------------

def setup_container():

    global _bootstrapped

    if _bootstrapped:
        return

    # providers
    load_providers("rpgbot.infrastructure.llm")
    load_providers("rpgbot.infrastructure.embeddings")

    register_llm()
    register_embeddings()

    # utilities
    container.register("semantic_cache", HierarchicalSemanticCache)
    container.register("hash", lambda: sha256_hash)

    container.register(
        "embed",
        lambda embedding_provider: embedding_provider.embed,
        singleton=True
    )

    # stores
    container.register("vector_store", MemoryMappedVectorStore)
    container.register("document_store", DocumentStore)
    container.register("token_store", TokenStore)
    container.register("metadata_store", MetadataStore)

    container.register(
        "feature_store",
        lambda: FeatureStore(CAMPAIGN_DIR)
    )

    # indexing
    container.register(
        "document_loader",
        lambda: DocumentLoader(CAMPAIGN_DIR)
    )

    container.register(
        "repository",
        lambda: DocumentRepository(CAMPAIGN_DIR / "vector.index")
    )

    container.register("embedding_indexer", EmbeddingIndexer)

    # query
    container.register(
        "query_classifier",
        QueryClassifier,
        singleton=True
    )

    # ranking
    container.register("stage1_ranker", Stage1Ranker)

    container.register(
        "stage2_ranker",
        lambda feature_store: Stage2Ranker(feature_store)
    )

    # clustering
    container.register("cluster_builder", ClusterBuilder)
    container.register("drift_detector", ClusterDriftDetector)
    container.register("cluster_manager", ClusterManager)

    # ANN
    container.register("ivf_builder", IVFBuilder)

    container.register(
        "ivf_router",
        lambda vector_store: IVFRouter(None, vector_store)
    )

    # vector index
    container.register("vector_index", build_vector_index_service)

    # retrieval
    container.register(
        "retrieval_engine",
        lambda vector_index: RetrievalEngine(vector_index)
    )

    # repositories
    container.register(
        "session_repository",
        lambda: AsyncJSONRepository(),
        singleton=True
    )

    _bootstrapped = True
