import time
import logging
from pathlib import Path

from rpgbot.infrastructure.embedding_cache import embed
from rpgbot.utils.json_store import load_json, save_json
from rpgbot.utils.vector_utils import vector_search

logger = logging.getLogger(__name__)


EVENT_FILE = Path("campaign/memory/events.json")
SESSION_FILE = Path("campaign/memory/sessions.json")
ARC_FILE = Path("campaign/memory/arcs.json")


async def log_event(text):

    events = load_json(EVENT_FILE, [])

    events.append({
        "timestamp": time.time(),
        "text": text,
        "vector": await embed(text)
    })

    save_json(EVENT_FILE, events[-100:])


def get_recent_events(limit=5):

    events = load_json(EVENT_FILE, [])

    return [e["text"] for e in events[-limit:]]


async def search_events(query, k=3):

    events = load_json(EVENT_FILE, [])

    return await vector_search(events, query, "text", k)


async def search_sessions(query, k=2):

    sessions = load_json(SESSION_FILE, [])

    return await vector_search(sessions, query, "summary", k)


async def search_arcs(query, k=2):

    arcs = load_json(ARC_FILE, [])

    return await vector_search(arcs, query, "summary", k)


async def hierarchical_search(query):

    return (
        await search_arcs(query)
        + await search_sessions(query)
        + await search_events(query)
    )

async def summarize_session(generate_narrative):

    events = load_json(EVENT_FILE, [])

    if not events:
        return

    text = "\n".join(e["text"] for e in events)

    summary = generate_narrative(
        f"Resuma os principais acontecimentos da sessão:\n{text}"
    )

    sessions = load_json(SESSION_FILE, [])

    sessions.append({
        "timestamp": time.time(),
        "summary": summary,
        "vector": await embed(summary)
    })

    save_json(SESSION_FILE, sessions[-50:])

    save_json(EVENT_FILE, [])