from pathlib import Path

import rpgbot.infrastructure.embedding_cache as cache


def reset_embedding_state():
    """
    Limpa completamente o estado interno do embedding cache.

    Mantido apenas em testes para preservar Clean Architecture.
    """

    cache._cache = None
    cache._index = None

    try:
        cache._ids.clear()
    except Exception:
        pass

    try:
        cache._lru_cache.clear()
    except Exception:
        pass

    try:
        cache._keyword_index.clear()
    except Exception:
        pass

    try:
        cache._graph_vectors.clear()
        cache._graph_edges.clear()
    except Exception:
        pass

    try:
        if cache.CACHE_PATH.exists():
            cache.CACHE_PATH.unlink()
    except Exception:
        pass
