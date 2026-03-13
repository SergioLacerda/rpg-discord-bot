import pytest
from unittest.mock import AsyncMock

from rpgbot.bootstrap import setup_container
from rpgbot.core.container import container
from rpgbot.adapters.storage.json_session_repository import (
    log_event,
    search_events,
    get_recent_events
)
from rpgbot.infrastructure.vector_index import VectorIndex
from rpgbot.usecases.retrieve_context import (
    index_campaign,
    search_context
)


# -----------------------------------------------------------
# Fake Vector Index determinístico
# -----------------------------------------------------------

class FakeVectorIndex:
    """
    VectorIndex fake determinístico e otimizado para testes.
    Pré-computa tokens e embeddings para evitar custo repetido.
    """

    def __init__(self, docs=None):

        raw_docs = docs or [
            {"text": "Stormy infiltrou o armazém secreto"},
            {"text": "Os jogadores chegaram ao porto"},
            {"text": "Um guarda patrulha a entrada"}
        ]

        self.docs = []

        for d in raw_docs:

            text = d["text"]
            tokens = set(text.lower().split())

            self.docs.append({
                "text": text,
                "tokens": tokens,
                "vector": self._embed_sync(text)
            })

    # ---------------------------------------------------------
    # embedding determinístico
    # ---------------------------------------------------------

    def _embed_sync(self, text):

        text = text.lower()

        if "stormy" in text:
            return (1.0, 0.0, 0.0)

        if "armazem" in text or "armazém" in text:
            return (0.0, 1.0, 0.0)

        return (0.0, 0.0, 1.0)

    async def embed(self, text):
        return self._embed_sync(text)

    # ---------------------------------------------------------
    # search rápida
    # ---------------------------------------------------------

    async def search(self, query, k=4):

        q_tokens = set(query.lower().split())

        scored = []

        for d in self.docs:

            score = len(q_tokens & d["tokens"])

            if score > 0:
                scored.append((score, d["text"]))

        scored.sort(reverse=True)

        return [t for _, t in scored[:k]]


@pytest.fixture
def fake_index(monkeypatch):

    container.reset()
    setup_container()  # ← restaura os serviços

    index = FakeVectorIndex()

    container.register("vector_index", lambda: index)

    monkeypatch.setattr(
        "rpgbot.infrastructure.embedding_client.remote_embed",
        AsyncMock(side_effect=index.embed)
    )

    return index


# -----------------------------------------------------------
# Tests
# -----------------------------------------------------------

@pytest.mark.asyncio
async def test_timeline(tmp_path, monkeypatch):

    timeline_file = tmp_path / "timeline.json"

    monkeypatch.setattr(
        "rpgbot.adapters.storage.json_session_repository.EVENT_FILE",
        timeline_file
    )

    async def fake_embed(text):
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(
        "rpgbot.infrastructure.embedding_client.remote_embed",
        AsyncMock(side_effect=fake_embed)
    )

    event_text = "Jogadores entraram no armazém secreto"

    await log_event(event_text)

    results = await search_events("armazém")

    assert len(results) >= 1
    assert event_text in results[0]


@pytest.mark.asyncio
async def test_search_context(fake_index):

    docs = await search_context("Stormy")

    assert docs
    assert "Stormy" in docs[0]


@pytest.mark.asyncio
async def test_incremental_index(tmp_path, monkeypatch):

    file = tmp_path / "test.md"
    file.write_text("conteudo")

    monkeypatch.setattr(
        "rpgbot.usecases.retrieve_context.CAMPAIGN_DIR",
        tmp_path
    )

    async def fake_embed(text):
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(
        "rpgbot.infrastructure.embedding_client.remote_embed",
        AsyncMock(side_effect=fake_embed)
    )

    docs = await index_campaign()

    assert len(docs) == 1