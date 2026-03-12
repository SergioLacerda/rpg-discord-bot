import re
from pathlib import Path
from collections import defaultdict

from rpgbot.utils import load_json, save_json


GRAPH_FILE = Path("campaign/memory/narrative_graph.json")

_graph_cache = None


def load_graph():

    global _graph_cache

    if _graph_cache is None:
        _graph_cache = load_json(GRAPH_FILE, {})

    return _graph_cache


def save_graph(graph):

    global _graph_cache

    _graph_cache = graph

    save_json(GRAPH_FILE, graph)


def extract_entities(text):

    words = re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", text)

    return list(set(words))


def update_graph_from_event(text):

    graph = load_graph()

    entities = extract_entities(text)

    for e in entities:

        node = graph.setdefault(e, {"links": []})

        for other in entities:

            if other == e:
                continue

            if other not in node["links"]:
                node["links"].append(other)

    save_graph(graph)


def related_entities(query, depth=1):

    graph = load_graph()

    entities = extract_entities(query)

    related = set()

    for e in entities:

        node = graph.get(e)

        if not node:
            continue

        related.update(node.get("links", []))

    return related