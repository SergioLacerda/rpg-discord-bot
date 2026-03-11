import time
import logging
from pathlib import Path

from src.infrastructure.embedding_cache import embed
from src.utils.json_store import load_json, save_json
from src.utils.vector_utils import vector_search

logger = logging.getLogger(__name__)


EVENT_FILE = Path("campaign/memory/events.json")
SESSION_FILE = Path("campaign/memory/sessions.json")
ARC_FILE = Path("campaign/memory/arcs.json")


def log_event(text):

    events = load_json(EVENT_FILE, [])

    events.append({
        "timestamp": time.time(),
        "text": text,
        "vector": embed(text)
    })

    save_json(EVENT_FILE, events[-100:])


def get_recent_events(limit=5):

    events = load_json(EVENT_FILE, [])

    return [e["text"] for e in events[-limit:]]


def search_events(query, k=3):

    events = load_json(EVENT_FILE, [])

    return vector_search(events, query, "text", k)


def search_sessions(query, k=2):

    sessions = load_json(SESSION_FILE, [])

    return vector_search(sessions, query, "summary", k)


def search_arcs(query, k=2):

    arcs = load_json(ARC_FILE, [])

    return vector_search(arcs, query, "summary", k)


def hierarchical_search(query):

    return (
        search_arcs(query)
        + search_sessions(query)
        + search_events(query)
    )