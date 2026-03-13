import math
import random
import heapq
from typing import List
from collections import Counter

from rpgbot.infrastructure.embedding_cache import embed
from rpgbot.campaign.memory.entity_alias_resolver import EntityAliasResolver


VECTOR_DIM = 1536
LSH_BITS = 12

_rng = random.Random(42)
_alias_resolver = None


PROJECTION = [_rng.gauss(0, 1) for _ in range(VECTOR_DIM)]

LSH_PLANES = [
    [_rng.gauss(0, 1) for _ in range(VECTOR_DIM)]
    for _ in range(LSH_BITS)
]


def get_alias_resolver():

    global _alias_resolver

    if _alias_resolver is None:
        _alias_resolver = EntityAliasResolver()

    return _alias_resolver


def cosine_similarity(a, b):

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def cosine_early_abandon(q_vec, d_vec, q_norm, d_norm, threshold):

    partial = 0.0

    remaining_q = q_norm

    for a, b in zip(q_vec, d_vec):

        prod = a * b

        partial += prod

        remaining_q -= abs(a)

        max_possible = partial + remaining_q * abs(d_norm)

        upper_bound = max_possible / (q_norm * d_norm)

        if upper_bound <= threshold:
            return None

    return partial / (q_norm * d_norm)


async def vector_search(items, query, field, k):

    resolver = get_alias_resolver()

    final_query = resolver.normalize(query)

    q_vec = await embed(final_query)

    q_norm = math.sqrt(sum(x * x for x in q_vec))

    if q_norm == 0:
        return []

    heap = []

    for item in items:

        vec = item["vector"]

        dot = 0.0

        for a, b in zip(q_vec, vec):
            dot += a * b

        norm_b = math.sqrt(sum(x * x for x in vec))

        if norm_b == 0:
            continue

        threshold = heap[0][0] if heap else -1

        score = cosine_early_abandon(
            q_vec,
            vec,
            q_norm,
            norm_b,
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

    if not results:

        q = final_query.lower()

        for item in items:

            text = item.get(field, "").lower()

            if q in text:

                results.append(item[field])

                if len(results) >= k:
                    break

    return results


def lsh_hash(vec: List[float]) -> str:

    bits = []

    for plane in LSH_PLANES:

        dot = 0.0

        for a, b in zip(vec, plane):
            dot += a * b

        bits.append("1" if dot >= 0 else "0")

    return "".join(bits)


def project(vec: List[float]) -> float:

    return sum(a * b for a, b in zip(vec, PROJECTION))


def keyword_score(query_tokens: List[str], doc_tokens: List[str]) -> float:

    if not doc_tokens:
        return 0.0

    tf = Counter(doc_tokens)

    score = 0.0

    for token in query_tokens:

        freq = tf.get(token, 0)

        if freq == 0:
            continue

        score += math.log(1 + freq)

    return score / (len(doc_tokens) + 1)