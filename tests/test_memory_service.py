import pytest

from src.services.session_memory import log_event, search_events, get_recent_events
from src.services.memory_service import (
    save_npc,
    get_npc,
    cosine_similarity,    
    build_context,
    load_index,
    index_campaign,
    search_context
)
from src.utils.json_store import load_json, save_json


def test_npc_persistence(tmp_path, monkeypatch):

    npc_file = tmp_path / "npc_database.json"

    monkeypatch.setattr(
        "src.services.memory_service.NPC_FILE",
        npc_file
    )

    save_npc("Stormy", "Agente infiltrador")

    npc = get_npc("Stormy")

    assert npc["description"] == "Agente infiltrador"
    assert "last_seen" in npc


def test_timeline(tmp_path, monkeypatch):

    timeline_file = tmp_path / "timeline.json"

    monkeypatch.setattr(
        "src.services.session_memory.EVENT_FILE",
        timeline_file
    )

    monkeypatch.setattr(
        "src.infrastructure.embedding_client.embed",
        lambda x: [1, 0, 0]
    )

    log_event("Jogadores entraram no armazém")

    events = search_events("armazém")

    assert "Jogadores entraram no armazém" in events[0]


def test_cosine_similarity():

    a = [1,0]
    b = [1,0]

    score = cosine_similarity(a,b)

    assert score == pytest.approx(1.0)


def test_json_roundtrip(tmp_path):

    file = tmp_path / "data.json"

    save_json(file, {"a":1})

    data = load_json(file, {})

    assert data["a"] == 1


def test_build_context(monkeypatch):

    monkeypatch.setattr(
        "src.services.memory_service.search_context",
        lambda q: ["doc1","doc2"]
    )

    monkeypatch.setattr(
        "src.services.memory_service.hierarchical_search",
        lambda q: ["mem1","mem2"]
    )

    context = build_context("teste")

    assert "doc1" in context
    assert "mem1" in context


def test_load_index_empty(tmp_path, monkeypatch):

    monkeypatch.setattr(
        "src.services.memory_service.VECTOR_FILE",
        tmp_path / "vectors.json"
    )

    monkeypatch.setattr(
        "src.services.memory_service.index_campaign",
        lambda : []
    )

    docs = load_index()

    assert docs == []


def test_search_context(monkeypatch):

    monkeypatch.setattr(
        "src.infrastructure.embedding_client.embed",
        lambda x: [1,0,0]
    )

    monkeypatch.setattr(
        "src.services.memory_service.load_index",
        lambda : [
            {"text":"doc1","vector":[1,0,0]},
            {"text":"doc2","vector":[0,1,0]}
        ]
    )

    docs = search_context("teste")

    assert docs[0] == "doc1"

def test_incremental_index(tmp_path, monkeypatch):

    file = tmp_path / "test.md"
    file.write_text("conteudo")

    monkeypatch.setattr(
        "src.services.memory_service.CAMPAIGN_DIR",
        tmp_path
    )

    monkeypatch.setattr(
        "src.infrastructure.embedding_client.embed",
        lambda x: [1,0,0]
    )

    docs = index_campaign()

    assert len(docs) == 1