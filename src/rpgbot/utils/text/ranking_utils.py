from collections import Counter
from pathlib import Path

from rpgbot.utils import load_json
from rpgbot.utils.text.normalize_utils import tokenize

EVENT_FILE = Path("campaign/memory/events.json")
NPC_FILE = Path("campaign/npc_database.json")
_context_cache = None
_entity_cache = None


def load_entities():
    """
    Carrega entidades da campanha (NPCs etc.)
    com cache em memória.
    """

    global _entity_cache

    if _entity_cache is None:

        db = load_json(NPC_FILE, {})

        _entity_cache = {k.lower() for k in db.keys()}

    return _entity_cache


def entity_boost(query: str, doc_text: str) -> float:
    """
    Aumenta score se a query menciona entidades
    presentes no documento.
    """

    entities = load_entities()

    if not entities:
        return 0.0

    q_tokens = set(tokenize(query))
    d_tokens = set(tokenize(doc_text))

    overlap = q_tokens & d_tokens & entities

    if not overlap:
        return 0.0

    return min(1.0, len(overlap) * 0.5)


def load_context_terms(limit=10):
    """
    Carrega termos dos eventos recentes da campanha.
    Usa cache em memória.
    """

    global _context_cache

    if _context_cache is not None:
        return _context_cache

    try:
        events = load_json(EVENT_FILE, [])
    except Exception:
        return Counter()

    tokens = []

    for e in events[-limit:]:
        tokens.extend(e.get("text", "").lower().split())

    _context_cache = Counter(tokens)

    return _context_cache


def contextual_score(doc_tokens):
    """
    Score baseado na frequência de termos do documento
    no contexto recente da campanha.
    """

    ctx = load_context_terms()

    if not ctx:
        return 0.0

    score = 0

    for token in doc_tokens:
        score += ctx.get(token, 0)

    return score / (len(doc_tokens) + 1)