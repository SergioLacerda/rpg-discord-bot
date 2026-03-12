import math
import random
from typing import List
from collections import Counter

from rpgbot.infrastructure.embedding_cache import embed


# configuração
VECTOR_DIM = 1536
LSH_BITS = 12

_rng = random.Random(42)

# vetor de projeção determinístico
PROJECTION = [_rng.gauss(0, 1) for _ in range(VECTOR_DIM)]

# hiperplanos gaussianos
LSH_PLANES = [
    [_rng.gauss(0, 1) for _ in range(VECTOR_DIM)]
    for _ in range(LSH_BITS)
]

def cosine_similarity(a, b):

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


async def vector_search(items, query, field, k):

    q_vec = await embed(query)

    scored = sorted(
        ((cosine_similarity(q_vec, i["vector"]), i) for i in items),
        reverse=True
    )

    return [i[field] for _, i in scored[:k]]


def lsh_hash(vec: List[float]) -> str:
    """
    Gera hash LSH usando hiperplanos aleatórios.

    Usa sign(random_projection) para aproximar
    cosine similarity.

    Retorna uma string binária como bucket id.
    """

    bits = []

    for plane in LSH_PLANES:

        dot = 0.0

        for a, b in zip(vec, plane):
            dot += a * b

        bits.append("1" if dot >= 0 else "0")

    return "".join(bits)


def project(vec: List[float]) -> float:
    """
    Projeta um vetor em um escalar usando
    random projection.

    Usado para ordenar vetores e permitir
    busca binária aproximada.
    """

    return sum(a * b for a, b in zip(vec, PROJECTION))


def keyword_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """
    Calcula um score lexical simples entre query e documento.

    Usa frequência logarítmica das palavras da query
    dentro do documento.

    Retorna valor normalizado entre 0 e ~1.
    """

    if not doc_tokens:
        return 0.0

    tf = Counter(doc_tokens)

    score = 0.0

    for token in query_tokens:

        freq = tf.get(token, 0)

        if freq == 0:
            continue

        # peso logarítmico (evita spam de palavras)
        score += math.log(1 + freq)

    return score / (len(doc_tokens) + 1)