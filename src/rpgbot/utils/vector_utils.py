import math

from rpgbot.infrastructure.embedding_cache import embed


def cosine_similarity(a, b):

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)

def vector_search(items, query, field, k):

    q_vec = embed(query)

    scored = sorted(
        ((cosine_similarity(q_vec, i["vector"]), i) for i in items),
        reverse=True
    )

    return [i[field] for _, i in scored[:k]]