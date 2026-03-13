from rpgbot.infrastructure.vector_index.index import VectorIndex
from rpgbot.infrastructure.vector_index.components import VectorIndexComponents


def build_vector_index(
    components,
    semantic_cache,
    document_loader,
    repository,
    embedding_indexer,
    feature_store,
    campaign_dir,
):

    return VectorIndex(
        components=components,
        semantic_cache=semantic_cache,
        document_loader=document_loader,
        repository=repository,
        embedding_indexer=embedding_indexer,
        campaign_dir=campaign_dir
    )