from collections import defaultdict


class NarrativeCausalityGraph:

    def __init__(self):

        self.graph = defaultdict(set)

    def add_edge(self, cause, effect):

        self.graph[cause].add(effect)

    def neighbors(self, doc_id):

        return self.graph.get(doc_id, set())

    def expand(self, doc_ids, depth=2):

        result = set(doc_ids)

        frontier = set(doc_ids)

        for _ in range(depth):

            next_nodes = set()

            for d in frontier:

                for n in self.graph.get(d, []):
                    next_nodes.add(n)

            result.update(next_nodes)

            frontier = next_nodes

        return list(result)