import time
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rpgbot.infrastructure.embedding_cache import embed, CACHE_PATH, _cache
from rpgbot.infrastructure.embedding_client import deterministic_vector

@pytest.fixture(autouse=True)
def clear_embedding_cache():
    cache_path = Path("campaign/memory/embedding_cache.json")
    if cache_path.exists():
        cache_path.unlink()
    yield
    if cache_path.exists():
        cache_path.unlink()

@pytest.mark.asyncio
async def test_embed_calls_remote_and_returns_vector(monkeypatch):
    fake_response = MagicMock()
    fake_response.data = [MagicMock()]
    fake_response.data[0].embedding = [0.1, 0.2, 0.3]

    fake_embeddings = MagicMock()
    fake_embeddings.create = AsyncMock(return_value=fake_response)

    fake_client = MagicMock()
    fake_client.embeddings = fake_embeddings

    monkeypatch.setattr(
        "rpgbot.infrastructure.embedding_client.get_client",
        AsyncMock(return_value=fake_client)
    )

    # Patch do remote_embed no namespace do cache (onde ele é usado)
    with patch("rpgbot.infrastructure.embedding_cache.remote_embed") as mock_remote:
        mock_remote.return_value = [0.1, 0.2, 0.3]

        vec = await embed("texto de teste qualquer único 987654")
        assert vec == [0.1, 0.2, 0.3], f"Vetor retornado foi: {vec}"
        assert mock_remote.call_count == 1

@pytest.mark.asyncio
async def test_embed_uses_cache_after_first_call(monkeypatch):
    fake_response = MagicMock()
    fake_response.data = [MagicMock()]
    fake_response.data[0].embedding = [0.4, 0.5, 0.6]

    fake_embeddings = MagicMock()
    fake_embeddings.create = AsyncMock(return_value=fake_response)

    fake_client = MagicMock()
    fake_client.embeddings = fake_embeddings

    monkeypatch.setattr(
        "rpgbot.infrastructure.embedding_client.get_client",
        AsyncMock(return_value=fake_client)
    )

    with patch("rpgbot.infrastructure.embedding_cache.remote_embed") as mock_remote:
        mock_remote.side_effect = lambda *a, **k: [0.4, 0.5, 0.6]

        unique_suffix = f"cache-test-{int(time.time_ns())}-{id(object())}"
        phrase = f"frase exclusiva teste cache {unique_suffix}"

        vec1 = await embed(phrase)
        assert vec1 == [0.4, 0.5, 0.6]
        assert mock_remote.call_count == 1

        vec2 = await embed(phrase)
        assert vec2 == [0.4, 0.5, 0.6]
        assert mock_remote.call_count == 1

@pytest.mark.asyncio
async def test_fallback_on_error():

    # limpa cache persistente
    global _cache
    _cache = None

    if CACHE_PATH.exists():
        CACHE_PATH.unlink()

    async def failing_remote(*args, **kwargs):
        raise Exception("Simulando API caída")

    with patch("rpgbot.infrastructure.embedding_cache.remote_embed") as mock_remote:

        mock_remote.side_effect = failing_remote

        vector = await embed("texto para testar fallback")

    expected = deterministic_vector("texto para testar fallback")

    assert len(vector) == 1536
    assert all(isinstance(x, float) for x in vector)
    assert vector == expected
    assert mock_remote.call_count == 1