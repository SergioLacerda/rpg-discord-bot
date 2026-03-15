import math
from typing import List, Tuple


# ---------------------------------------------------------
# basic vector ops
# ---------------------------------------------------------

def dot(a: List[float], b: List[float]) -> float:
    """Produto escalar otimizado."""
    total = 0.0
    la = len(a)

    for i in range(la):
        total += a[i] * b[i]

    return total


def l2_norm(v: List[float]) -> float:
    """Norma L2."""
    total = 0.0

    for x in v:
        total += x * x

    return math.sqrt(total)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity otimizada."""

    dot_val = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for i in range(len(a)):

        x = a[i]
        y = b[i]

        dot_val += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_val / (math.sqrt(norm_a) * math.sqrt(norm_b))


# ---------------------------------------------------------
# optimized cosine with early abandon
# ---------------------------------------------------------

def cosine_early_abandon(
    q_vec: List[float],
    d_vec: List[float],
    q_norm: float,
    d_norm: float,
    threshold: float
) -> float | None:
    """
    Cosine similarity com early abandon.

    Pode abortar cálculo se score máximo possível
    não superar threshold atual.
    """

    partial = 0.0
    remaining = 0.0

    for x in q_vec:
        remaining += abs(x)

    for i in range(len(q_vec)):

        a = q_vec[i]
        b = d_vec[i]

        partial += a * b
        remaining -= abs(a)

        max_possible = partial + remaining * abs(d_norm)

        upper_bound = max_possible / (q_norm * d_norm)

        if upper_bound <= threshold:
            return None

    return partial / (q_norm * d_norm)


# ---------------------------------------------------------
# top-k similarity
# ---------------------------------------------------------

def top_k_cosine(
    query: List[float],
    docs: List[List[float]],
    k: int
) -> List[Tuple[float, int]]:
    """
    Retorna top-k similaridades.
    """

    q_norm = l2_norm(query)

    results = []

    for i, vec in enumerate(docs):

        score = cosine_similarity(query, vec)

        if len(results) < k:
            results.append((score, i))
            results.sort()

        else:

            if score > results[0][0]:
                results[0] = (score, i)
                results.sort()

    return sorted(results, reverse=True)
