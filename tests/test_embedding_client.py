# tests/test_embedding_client.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from rpgbot.infrastructure.embedding_cache import embed
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
    # Mock do cliente fake completo
    fake_response = MagicMock()
    fake_response.data = [MagicMock()]
    fake_response.data[0].embedding = [0.1, 0.2, 0.3]

    fake_embeddings = MagicMock()
    fake_embeddings.create = AsyncMock(return_value=fake_response)

    fake_client = MagicMock()
    fake_client.embeddings = fake_embeddings

    # Mock get_client → retorna fake_client
    monkeypatch.setattr(
        "rpgbot.infrastructure.embedding_client.get_client",
        AsyncMock(return_value=fake_client)   # async porque get_client pode ser await
    )

    # Reforço: mock direto do remote_embed (caso o cache chame ele)
    monkeypatch.setattr(
        "rpgbot.infrastructure.embedding_client.remote_embed",
        AsyncMock(return_value=[0.1, 0.2, 0.3])
    )

    vec = await embed("texto de teste qualquer único 987654")
    assert vec == [0.1, 0.2, 0.3], f"Vetor retornado foi: {vec}"


@pytest.mark.asyncio
async def test_embed_uses_cache_after_first_call(monkeypatch):
    from unittest.mock import patch

    # Mock do get_client (opcional, mas bom manter)
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

    call_count = [0]

    async def counting_remote(*args, **kwargs):
        call_count[0] += 1
        return [0.4, 0.5, 0.6]

    # Usamos patch com contexto → garante que o mock seja aplicado no momento certo
    with patch("rpgbot.infrastructure.embedding_cache.remote_embed") as mock_remote:
        mock_remote.side_effect = counting_remote

        import time
        unique_suffix = f"cache-test-{int(time.time_ns())}-{id(object())}"
        phrase = f"frase exclusiva teste cache {unique_suffix}"

        # Primeira chamada
        vec1 = await embed(phrase)
        assert vec1 == [0.4, 0.5, 0.6]
        assert call_count[0] == 1

        # Segunda chamada
        vec2 = await embed(phrase)
        assert vec2 == [0.4, 0.5, 0.6]
        assert call_count[0] == 1


@pytest.mark.asyncio
async def test_fallback_on_error(monkeypatch):
    # Força erro no remote_embed
    async def failing_remote(*args, **kwargs):
        raise Exception("Simulando API caída")

    monkeypatch.setattr(
        "rpgbot.infrastructure.embedding_client.remote_embed",
        AsyncMock(side_effect=failing_remote)
    )

    vector = await embed("texto para testar fallback")
    expected = deterministic_vector("texto para testar fallback")
    assert len(vector) == 1536
    assert all(isinstance(x, float) for x in vector)
    assert vector == expected, "Fallback não retornou o vetor determinístico esperado"