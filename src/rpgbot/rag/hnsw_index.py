import random
from collections import defaultdict

from rpgbot.utils.vector.vector_math import cosine_similarity


class HNSWIndex:

    def __init__(self, docs, M=6, ef=20):

        self.M = M          # número de vizinhos
        self.ef = ef        # largura da busca

        self.docs = docs
        self.layers = []
        self.graph = defaultdict(list)

        self.entry_point = None

        self._build()

    # ---------------------------------------------------------
    # construção
    # ---------------------------------------------------------

    def _random_level(self):

        level = 0
        while random.random() < 0.5:
            level += 1
        return level

    def _build(self):

        for doc in self.docs:

            level = self._random_level()

            while len(self.layers) <= level:
                self.layers.append([])

            for l in range(level + 1):
                self.layers[l].append(doc)

            if self.entry_point is None:
                self.entry_point = doc
                continue

            neighbors = self._search_layer(doc["vector"], self.entry_point)

            for n in neighbors[:self.M]:

                self.graph[doc["id"]].append(n)
                self.graph[id(n)].append(doc)

    # ---------------------------------------------------------
    # busca em camada
    # ---------------------------------------------------------

    def _search_layer(self, q_vec, entry):

        visited = set()
        candidates = [entry]
        best = [entry]

        while candidates:

            current = candidates.pop()

            for neighbor in self.graph[id(current)]:

                if id(neighbor) in visited:
                    continue

                visited.add(id(neighbor))

                score = cosine_similarity(q_vec, neighbor["vector"])

                if len(best) < self.ef or score > cosine_similarity(q_vec, best[-1]["vector"]):
                    best.append(neighbor)
                    best.sort(
                        key=lambda d: cosine_similarity(q_vec, d["vector"]),
                        reverse=True
                    )

                    if len(best) > self.ef:
                        best.pop()

                    candidates.append(neighbor)

        return best

    # ---------------------------------------------------------
    # busca principal
    # ---------------------------------------------------------

    def search(self, q_vec, k=20):

        if not self.entry_point:
            return []

        entry = self.entry_point

        for layer in reversed(self.layers):

            best = self._search_layer(q_vec, entry)

            if best:
                entry = best[0]

        result = self._search_layer(q_vec, entry)

        return result[:k]