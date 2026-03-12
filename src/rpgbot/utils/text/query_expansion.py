from rpgbot.utils import load_json
from rpgbot.utils.text.normalize_utils import tokenize
from collections import Counter
from pathlib import Path

EVENT_FILE = Path("campaign/memory/events.json")


def expand_query(query, limit=5):

    try:
        events = load_json(EVENT_FILE, [])
    except Exception:
        return query

    tokens = []

    for e in events[-10:]:
        tokens.extend(tokenize(e.get("text", "")))

    if not tokens:
        return query

    freq = Counter(tokens)

    extra = [t for t, _ in freq.most_common(limit)]

    return query + " " + " ".join(extra)