import time
import logging
from pathlib import Path

from rpgbot.core.runtime_state import bump_event_version
from rpgbot.infrastructure.embedding_cache import embed
from rpgbot.infrastructure.narrative_graph import update_graph_from_event
from rpgbot.utils.vector.vector_utils import vector_search
from rpgbot.utils import load_json, save_json
from rpgbot.utils.text.normalize_utils import tokenize
from rpgbot.campaign.memory.entity_alias_resolver import EntityAliasResolver

logger = logging.getLogger(__name__)
alias_resolver = EntityAliasResolver()

EVENT_FILE = Path("campaign/memory/events.json")
SESSION_FILE = Path("campaign/memory/sessions.json")
ARC_FILE = Path("campaign/memory/arcs.json")


# ------------------------------------------------------------------
# utilidades
# ------------------------------------------------------------------

def _ensure_file(path):

    if not path.exists():
        save_json(path, [])


def _load_events():
    _ensure_file(EVENT_FILE)
    return load_json(EVENT_FILE, [])


def _load_sessions():
    _ensure_file(SESSION_FILE)
    return load_json(SESSION_FILE, [])


def _load_arcs():
    _ensure_file(ARC_FILE)
    return load_json(ARC_FILE, [])


# ------------------------------------------------------------------
# eventos
# ------------------------------------------------------------------

async def log_event(text):

    events = _load_events()

    query = alias_resolver.normalize(text)

    vector = await embed(query)

    event = {
        "timestamp": time.time(),
        "text": text,
        "tokens": tokenize(text),
        "vector": vector
    }

    events.append(event)

    # atualizar grafo narrativo
    try:
        update_graph_from_event(text)
    except Exception:
        logger.exception("Falha ao atualizar narrative graph")

    # limitar histórico
    events = events[-200:]

    save_json(EVENT_FILE, events)

    bump_event_version()

    logger.debug("Evento registrado")


def get_recent_events(limit=5):

    events = _load_events()

    return [e["text"] for e in events[-limit:]]


# ------------------------------------------------------------------
# busca semântica
# ------------------------------------------------------------------

async def search_events(query, k=3):

    events = _load_events()

    if not events:
        return []

    return await vector_search(events, query, "text", k)


async def search_sessions(query, k=2):

    sessions = _load_sessions()

    if not sessions:
        return []

    return await vector_search(sessions, query, "summary", k)


async def search_arcs(query, k=2):

    arcs = _load_arcs()

    if not arcs:
        return []

    return await vector_search(arcs, query, "summary", k)


async def hierarchical_search(query):

    arcs = await search_arcs(query)
    sessions = await search_sessions(query)
    events = await search_events(query)

    return arcs + sessions + events


# ------------------------------------------------------------------
# sumarização de sessão
# ------------------------------------------------------------------

def compress_events(events):

    size = 0
    selected = []

    for e in reversed(events):

        txt = e["text"]

        size += len(txt)

        if size > MAX_EVENT_CHARS:
            break

        selected.append(txt)

    return "\n".join(reversed(selected))


async def summarize_session(generate_narrative):

    events = _load_events()

    if not events:
        logger.info("Nenhum evento para resumir")
        return

    # -------------------------------------------------
    # TOKEN REDUCTION
    # -------------------------------------------------

    text = compress_events(events)

    prompt = f"Resuma os principais acontecimentos da sessão:\n{text}"

    summary = await generate_narrative(prompt)

    sessions = _load_sessions()

    session_record = {
        "timestamp": time.time(),
        "summary": summary,
        "tokens": tokenize(summary),
        "vector": await embed(summary)
    }

    sessions.append(session_record)

    sessions = sessions[-100:]

    save_json(SESSION_FILE, sessions)

    save_json(EVENT_FILE, [])

    logger.info("Sessão resumida e arquivada")