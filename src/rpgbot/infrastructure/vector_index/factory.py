from pathlib import Path

from rpgbot.infrastructure.vector_index.index import VectorIndex
from rpgbot.infrastructure.vector_index.components import VectorIndexComponents
from rpgbot.infrastructure.vector_index.stores.mmap_vector_store import MMapVectorStore


def build_vector_index(
    components,
    semantic_cache,
    document_loader,
    repository,
    embedding_indexer,
    feature_store,
    campaign_dir,
):

    # ---------------------------------------------------------
    # memory mapped vector store
    # ---------------------------------------------------------

    vector_path = Path(campaign_dir) / "vector_store.memmap"

    vector_dim = getattr(components, "vector_dim", 768)

    components.vector_store = MMapVectorStore(
        path=vector_path,
        dim=vector_dim,
        capacity=1_000_000
    )

    # ---------------------------------------------------------
    # build index
    # ---------------------------------------------------------

    return VectorIndex(
        components=components,
        semantic_cache=semantic_cache,
        document_loader=document_loader,
        repository=repository,
        embedding_indexer=embedding_indexer,
        campaign_dir=campaign_dir
    )