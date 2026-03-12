import pytest
from unittest.mock import AsyncMock, patch

from rpgbot.services.session_memory import log_event, search_events, get_recent_events
from rpgbot.services.memory_service import (
    save_npc,
    get_npc,
    cosine_similarity,    
    hierarchical_context,
    load_index,
    index_campaign,
    search_context
)
from rpgbot.utils.json_store import load_json, save_json


@pytest.mark.asyncio
async def test_timeline(tmp_path, monkeypatch):
    timeline_file = tmp_path / "timeline.json"
    monkeypatch.setattr("rpgbot.services.session_memory.EVENT_FILE", timeline_file)

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
async def test_search_context(monkeypatch):
    async def fake_embed(text):
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(
        "rpgbot.infrastructure.embedding_client.remote_embed",
        AsyncMock(side_effect=fake_embed)
    )

    # Mock do load_index (retorna dados prontos)
    monkeypatch.setattr(
        "rpgbot.services.memory_service.load_index",
        lambda: [
            {"text": "doc1", "vector": [1.0, 0.0, 0.0]},
            {"text": "doc2", "vector": [0.0, 1.0, 0.0]}
        ]
    )

    docs = await search_context("teste")

    assert docs
    assert docs[0] == "doc1"


@pytest.mark.asyncio
async def test_incremental_index(tmp_path, monkeypatch):
    file = tmp_path / "test.md"
    file.write_text("conteudo")

    monkeypatch.setattr(
        "rpgbot.services.memory_service.CAMPAIGN_DIR",
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