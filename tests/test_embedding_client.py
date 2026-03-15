import asyncio
import time
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rpgbot.infrastructure.embedding_cache import (
    embed,
    CACHE_PATH,
    _cache,
    _lru_cache,
    _keyword_index,
    _graph_vectors,
    _graph_edges,
)

from rpgbot.infrastructure.embedding_client import deterministic_vector


# ---------------------------------------------------------
# limpar todos os caches antes de cada teste
# ---------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_embedding_state():

    global _cache

    _cache = None

    _lru_cache.clear()
    _keyword_index.clear()
    _graph_vectors.clear()
    _graph_edges.clear()

    if CACHE_PATH.exists():
        CACHE_PATH.unlink()

    yield

    if CACHE_PATH.exists():
        CACHE_PATH.unlink()


# ---------------------------------------------------------
# remote embed chamado
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_calls_remote_and_returns_vector():

    with patch("rpgbot.infrastructure.embedding_cache.remote_embed") as mock_remote:

        mock_remote.return_value = [0.1, 0.2, 0.3]

        vec = await embed("texto de teste qualquer único 987654")

        assert vec == [0.1, 0.2, 0.3]
        assert mock_remote.call_count == 1


# ---------------------------------------------------------
# cache persistente
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_uses_cache_after_first_call():

    with patch("rpgbot.infrastructure.embedding_cache.remote_embed") as mock_remote:

        mock_remote.return_value = [0.4, 0.5, 0.6]

        unique_suffix = f"cache-test-{time.time_ns()}"
        phrase = f"frase exclusiva teste cache {unique_suffix}"

        vec1 = await embed(phrase)

        assert vec1 == [0.4, 0.5, 0.6]
        assert mock_remote.call_count == 1

        vec2 = await embed(phrase)

        assert vec2 == [0.4, 0.5, 0.6]
        assert mock_remote.call_count == 1


# ---------------------------------------------------------
# fallback determinístico
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_on_error():

    async def failing_remote(*args, **kwargs):
        raise Exception("Simulando API caída")

    with patch(
        "rpgbot.infrastructure.embedding_cache.remote_embed",
        side_effect=failing_remote,
    ) as mock_remote:

        text = "texto para testar fallback"

        vector = await embed(text)

        expected = deterministic_vector(text)

        assert vector == expected
        assert len(vector) == 1536
        assert all(isinstance(x, float) for x in vector)

        assert mock_remote.call_count == 1


# ---------------------------------------------------------
# deduplicação concorrente
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_embedding():

    calls = {"n": 0}

    async def fake_remote(x):

        calls["n"] += 1

        await asyncio.sleep(0.1)

        return [1.0] * 1536

    with patch(
        "rpgbot.infrastructure.embedding_cache.remote_embed",
        side_effect=fake_remote,
    ):

        results = await asyncio.gather(
            embed("texto concorrente único 123"),
            embed("texto concorrente único 123"),
            embed("texto concorrente único 123"),
        )

    assert all(r == results[0] for r in results)

    # deduplicator deve chamar remote apenas 1 vez
    assert calls["n"] == 1
