from rpgbot.campaign.memory.entity_alias_resolver import EntityAliasResolver
from rpgbot.core.container import container
from rpgbot.core.campaign_context import CampaignContext
from rpgbot.infrastructure.embedding_cache import embed
from rpgbot.infrastructure.vector_index import VectorIndex
from rpgbot.rag.semantic_cache import HierarchicalSemanticCache
from rpgbot.rag.vector_ann_index import VectorANNIndex
from rpgbot.usecases.retrieval_engine import RetrievalEngine

_initialized = False

def setup_container():

    global _initialized

    if _initialized:
        return

    container.register("embed", lambda: embed)
    container.register("alias_resolver", EntityAliasResolver)
    container.register("semantic_cache", HierarchicalSemanticCache)
    container.register("ann_index_factory", VectorANNIndex)
    container.register("vector_index", VectorIndex)
    container.register("retrieval_engine", RetrievalEngine)
    container.register("campaign_context", CampaignContext)

    _initialized = True