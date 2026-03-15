import asyncio
import json
import logging
from collections import OrderedDict, Counter, defaultdict
from pathlib import Path
from typing import List, Dict
import math
import hnswlib

from rpgbot.core.container import container
from rpgbot.infrastructure.embedding_client import remote_embed, deterministic_vector

from rpgbot.utils import embedding_key
from rpgbot.utils.concurrency.deduplicate_async import InflightDeduplicator
from rpgbot.utils.vector.vector_math import cosine_similarity

GRAPH_NEIGHBORS = 6
GRAPH_THRESHOLD = 0.92

_graph_vectors = []
_graph_keys = []
_graph_edges = defaultdict(set)

LRU_SIZE = 512

_lru_cache: OrderedDict[str, List[float]] = OrderedDict()
_keyword_index: dict[str, set] = {}

logger = logging.getLogger(__name__)

CACHE_PATH = Path("campaign/memory/embedding_cache.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

SIMILARITY_THRESHOLD = 0.92
GRAPH_THRESHOLD = 0.88

deduplicator = InflightDeduplicator()

_cache: Dict = {}
_index: hnswlib.Index | None = None
_ids: List[str] = []
_graph: Dict[str, List[str]] = {}

DIMENSION = 1536


# ---------------------------------------------------------
# provider fingerprint
# ---------------------------------------------------------

def _get_embedding_fingerprint():

    try:
        provider = container.resolve("embedding_provider")
    except Exception:
        return "unknown"

    name = provider.__class__.__name__

    model = getattr(provider, "model_name", None) \
        or getattr(provider, "model", None) \
        or "default"

    dim = getattr(provider, "dimension", DIMENSION)

    return f"{name}:{model}:{dim}"


# ---------------------------------------------------------
# cache persistence
# ---------------------------------------------------------

def _load_cache_sync():

    if CACHE_PATH.exists():

        try:
            return json.loads(CACHE_PATH.read_text())

        except Exception as e:
            logger.warning(f"Erro ao carregar cache: {e}")

    return {}


def _save_cache_sync(data):

    try:
        CACHE_PATH.write_text(json.dumps(data))

    except Exception as e:
        logger.warning(f"Erro ao salvar cache: {e}")


async def load_cache():

    global _cache

    if not _cache:
        _cache = await asyncio.to_thread(_load_cache_sync)

    return _cache


async def save_cache():

    await asyncio.to_thread(_save_cache_sync, _cache)


def _cosine(a, b):

    dot = 0.0
    na = 0.0
    nb = 0.0

    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y

    if na == 0 or nb == 0:
        return 0.0

    return dot / math.sqrt(na * nb)


def _graph_lookup(vec):

    if not _graph_vectors:
        return None

    best = None
    best_score = 0.0

    for i, gvec in enumerate(_graph_vectors):

        s = _cosine(vec, gvec)

        if s > best_score:
            best_score = s
            best = i

    if best_score >= GRAPH_THRESHOLD:
        return _graph_vectors[best]

    return None


def _graph_add(vec, key):

    idx = len(_graph_vectors)

    _graph_vectors.append(vec)
    _graph_keys.append(key)

    neighbors = []

    for i, other in enumerate(_graph_vectors[:-1]):

        score = _cosine(vec, other)

        if score > GRAPH_THRESHOLD:
            neighbors.append((score, i))

    neighbors.sort(reverse=True)

    for _, n in neighbors[:GRAPH_NEIGHBORS]:

        _graph_edges[idx].add(n)
        _graph_edges[n].add(idx)

# ---------------------------------------------------------
# ANN index
# ---------------------------------------------------------

def _init_index(dim):

    global _index

    if _index:
        return

    _index = hnswlib.Index(space="cosine", dim=dim)

    _index.init_index(
        max_elements=200000,
        ef_construction=200,
        M=16
    )

    _index.set_ef(50)


def _rebuild_index(cache):

    global _ids

    if not cache:
        return

    first = next(iter(cache.values()))

    dim = len(first["vector"])

    _init_index(dim)

    vectors = []
    ids = []

    for i, (k, v) in enumerate(cache.items()):

        vec = v.get("vector")

        if not vec:
            continue

        vectors.append(vec)
        ids.append(i)
        _ids.append(k)

    if vectors:
        _index.add_items(vectors, ids)


# ---------------------------------------------------------
# semantic ANN lookup
# ---------------------------------------------------------

def _semantic_lookup(vector):

    if not _index or _index.get_current_count() == 0:
        return None

    labels, distances = _index.knn_query(vector, k=1)

    label = labels[0][0]
    dist = distances[0][0]

    similarity = 1 - dist

    if similarity >= SIMILARITY_THRESHOLD:

        key = _ids[label]

        entry = _cache.get(key)

        if entry:
            return entry["vector"]

    return None


# ---------------------------------------------------------
# graph builder
# ---------------------------------------------------------

def _update_graph(new_key, vector):

    for key, entry in _cache.items():

        if key == new_key:
            continue

        other = entry["vector"]

        try:
            sim = cosine_similarity(vector, other)
        except Exception:
            continue

        if sim >= GRAPH_THRESHOLD:

            _graph.setdefault(new_key, []).append(key)
            _graph.setdefault(key, []).append(new_key)


# ---------------------------------------------------------
# graph context expansion
# ---------------------------------------------------------

def get_related_embeddings(key: str, depth=2):

    visited = set()
    frontier = [key]

    for _ in range(depth):

        new_frontier = []

        for node in frontier:

            if node in visited:
                continue

            visited.add(node)

            for neighbor in _graph.get(node, []):
                if neighbor not in visited:
                    new_frontier.append(neighbor)

        frontier = new_frontier

    return [ _cache[k]["vector"] for k in visited if k in _cache ]


# ---------------------------------------------------------
# main embedding function
# ---------------------------------------------------------


def _lru_get(key):

    vec = _lru_cache.get(key)

    if vec is not None:
        _lru_cache.move_to_end(key)

    return vec


def _lru_put(key, vec):

    _lru_cache[key] = vec
    _lru_cache.move_to_end(key)

    if len(_lru_cache) > LRU_SIZE:
        _lru_cache.popitem(last=False)


def _keyword_tokens(text):

    return [t for t in text.lower().split() if len(t) > 3]


def _keyword_lookup(text):

    tokens = _keyword_tokens(text)

    if not tokens:
        return None

    scores = Counter()

    for token in tokens:

        if token not in _keyword_index:
            continue

        for key in _keyword_index[token]:
            scores[key] += 1

    if not scores:
        return None

    best_key, score = scores.most_common(1)[0]

    if score < 2:
        return None

    return best_key


def _keyword_index_add(text, key):

    for token in _keyword_tokens(text):

        bucket = _keyword_index.setdefault(token, set())

        bucket.add(key)


async def embed(text: str) -> List[float]:

    if not text.strip():
        return deterministic_vector(text)

    normalized = " ".join(text.lower().split())

    fingerprint = _get_embedding_fingerprint()

    key = embedding_key(f"{fingerprint}:{normalized}")

    # --------------------------------------------------
    # L0 keyword shortcut
    # --------------------------------------------------

    keyword_hit = _keyword_lookup(normalized)

    if keyword_hit:

        cache = await load_cache()

        entry = cache.get(keyword_hit)

        if entry:
            vec = entry["vector"]
            _lru_put(key, vec)
            return vec

    # --------------------------------------------------
    # L1 LRU
    # --------------------------------------------------

    vec = _lru_get(key)

    if vec is not None:
        return vec

    # --------------------------------------------------
    # L2 persistent cache
    # --------------------------------------------------

    cache = await load_cache()

    if _index is None and cache:
        _rebuild_index(cache)

    if key in cache:

        vec = cache[key]["vector"]

        _lru_put(key, vec)

        return vec

    # --------------------------------------------------
    # geração deduplicada
    # --------------------------------------------------

    async def _generate():

        try:

            vec = await asyncio.wait_for(
                remote_embed(text),
                timeout=20
            )

        except asyncio.CancelledError:
            raise

        except Exception as e:

            logger.warning(
                f"[embed] fallback determinístico ativado: {e}"
            )

            vec = deterministic_vector(text)

        # --------------------------------------------------
        # L2.5 Graph semantic memory
        # --------------------------------------------------

        graph_hit = _graph_lookup(vec)

        if graph_hit:
            _lru_put(key, graph_hit)
            return graph_hit

        # --------------------------------------------------
        # ANN semantic cache
        # --------------------------------------------------

        semantic_hit = _semantic_lookup(vec)

        if semantic_hit:
            _lru_put(key, semantic_hit)
            return semantic_hit

        # --------------------------------------------------
        # persist cache
        # --------------------------------------------------

        cache[key] = {
            "text": normalized,
            "vector": vec
        }

        idx = len(_ids)

        _ids.append(key)

        if _index:
            _index.add_items([vec], [idx])

        _keyword_index_add(normalized, key)

        _graph_add(vec, key)

        _lru_put(key, vec)

        return vec

    return await deduplicator.run(key, _generate)




