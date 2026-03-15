import math
import heapq
import random
from typing import List

from rpgbot.campaign.memory.entity_alias_resolver import EntityAliasResolver

from rpgbot.utils.vector.vector_math import (
    cosine_early_abandon,
    l2_norm,
)

# ---------------------------------------------------------
# config
# ---------------------------------------------------------

VECTOR_DIM = 1536
LSH_BITS = 12
PROJECTION_DIM = 64
CANDIDATE_MULTIPLIER = 4

_rng = random.Random(42)
_alias_resolver = None


# ---------------------------------------------------------
# random projections
# ---------------------------------------------------------

PROJECTION = [_rng.gauss(0, 1) for _ in range(VECTOR_DIM)]

LSH_PLANES = [
    [_rng.gauss(0, 1) for _ in range(VECTOR_DIM)]
    for _ in range(LSH_BITS)
]


# ---------------------------------------------------------
# resolver
# ---------------------------------------------------------

def get_alias_resolver():

    global _alias_resolver

    if _alias_resolver is None:
        _alias_resolver = EntityAliasResolver()

    return _alias_resolver


# ---------------------------------------------------------
# projection
# ---------------------------------------------------------

def project(vec: List[float]) -> float:

    dot = 0.0

    for a, b in zip(vec, PROJECTION):
        dot += a * b

    return dot


# ---------------------------------------------------------
# lsh hash
# ---------------------------------------------------------

def lsh_hash(vec: List[float]) -> str:

    bits = []

    for plane in LSH_PLANES:

        dot = 0.0

        for a, b in zip(vec, plane):
            dot += a * b

        bits.append("1" if dot >= 0 else "0")

    return "".join(bits)


# ---------------------------------------------------------
# vector search
# ---------------------------------------------------------

async def vector_search(items, query, field, k, embed_batch):

    resolver = get_alias_resolver()

    final_query = resolver.normalize(query)

    # -----------------------------------------
    # batch embedding
    # -----------------------------------------

    vectors = await remote_embed_batch([final_query])

    q_vec = vectors[0]

    q_norm = l2_norm(q_vec)

    if q_norm == 0:
        return []

    # -----------------------------------------
    # projection pruning
    # -----------------------------------------

    q_proj = project(q_vec)

    projections = []

    for item in items:

        vec = item["vector"]

        proj = project(vec)

        dist = abs(q_proj - proj)

        projections.append((dist, item))

    projections.sort(key=lambda x: x[0])

    candidate_count = min(len(projections), k * CANDIDATE_MULTIPLIER)

    candidates = [i[1] for i in projections[:candidate_count]]

    # -----------------------------------------
    # similarity search
    # -----------------------------------------

    heap = []

    for item in candidates:

        vec = item["vector"]

        d_norm = l2_norm(vec)

        if d_norm == 0:
            continue

        threshold = heap[0][0] if heap else -1

        score = cosine_early_abandon(
            q_vec,
            vec,
            q_norm,
            d_norm,
            threshold
        )

        if score is None:
            continue

        if len(heap) < k:

            heapq.heappush(heap, (score, item))

        else:

            heapq.heappushpop(heap, (score, item))

    heap.sort(reverse=True)

    results = [i[field] for _, i in heap]

    # -----------------------------------------
    # fallback textual
    # -----------------------------------------

    if not results:

        q = final_query.lower()

        for item in items:

            text = item.get(field, "").lower()

            if q in text:

                results.append(item[field])

                if len(results) >= k:
                    break

    return results


# ---------------------------------------------------------
# keyword scoring
# ---------------------------------------------------------

def keyword_score(query_tokens: List[str], doc_tokens: List[str]) -> float:

    if not doc_tokens:
        return 0.0

    from collections import Counter

    tf = Counter(doc_tokens)

    score = 0.0

    for token in query_tokens:

        freq = tf.get(token, 0)

        if freq == 0:
            continue

        score += math.log(1 + freq)

    return score / (len(doc_tokens) + 1)
