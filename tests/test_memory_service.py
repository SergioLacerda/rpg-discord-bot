import pytest
from unittest.mock import AsyncMock

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

class FakeVectorIndex(VectorIndex):

    def __init__(self):

        super().__init__()

        self.docs = [
            {"text": "Stormy invadiu Aurora", "vector": [1.0, 0.0, 0.0], "proj": 0.1},
            {"text": "NovaCorp investiga o caso", "vector": [0.8, 0.1, 0.0], "proj": 0.2},
            {"text": "Guardas patrulham o armazém", "vector": [0.0, 1.0, 0.0], "proj": 0.3},
        ]

        self.projections = [d["proj"] for d in self.docs]
        self.lsh_buckets = {}

    async def embed(self, text):
        return [1.0, 0.0, 0.0]


@pytest.fixture
def fake_index(monkeypatch):

    index = FakeVectorIndex()

    monkeypatch.setattr(
        "rpgbot.infrastructure.embedding_client.remote_embed",
        AsyncMock(side_effect=index.embed)
    )

    # substitui índice padrão usado por search_context
    monkeypatch.setattr(
        "rpgbot.usecases.retrieve_context._default_index",
        index
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