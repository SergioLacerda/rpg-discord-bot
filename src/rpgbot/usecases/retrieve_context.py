from pathlib import Path

from rpgbot.usecases.retrieval_engine import RetrievalEngine


CAMPAIGN_DIR = Path("campaign")

_engine = RetrievalEngine()

# compatibilidade com testes antigos
_default_index = _engine.index


async def search_context(query, k=4):

    if _default_index is not _engine.index:
        return await _default_index.search(query, k=k)

    return await _engine.search(query, k)


async def hierarchical_context(query, k=4):
    return await search_context(query, k)


async def index_campaign():
    _default_index.load()
    return _default_index.docs