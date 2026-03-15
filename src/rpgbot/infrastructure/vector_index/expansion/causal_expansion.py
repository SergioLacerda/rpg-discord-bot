
class CausalExpansion:
    """
    Expande candidatos usando relações de causalidade entre documentos/eventos.
    """

    priority = 50
    min_candidates = 1

    def __init__(
        self,
        max_expansion=20,
        depth=1,
        per_doc_limit=5,
        bidirectional=True
    ):

        self.max_expansion = max_expansion
        self.depth = depth
        self.per_doc_limit = per_doc_limit
        self.bidirectional = bidirectional


    def _neighbors(self, graph, doc_id):

        neighbors = set(graph.get(doc_id, []))

        if self.bidirectional and hasattr(graph, "reverse"):
            neighbors.update(graph.reverse.get(doc_id, []))

        return neighbors


    def run(self, ctx, candidates):

        index = ctx.index

        causality_graph = getattr(index, "causality_graph", None)

        if not causality_graph:
            return candidates

        doc_lookup = index.document_store.get

        expanded = []
        seen = {d["id"] for d in candidates}

        max_expansion = self.max_expansion

        frontier = [d["id"] for d in candidates]

        depth = self.depth

        for _ in range(depth):

            next_frontier = []

            for doc_id in frontier:

                neighbors = self._neighbors(causality_graph, doc_id)

                count = 0

                for rid in neighbors:

                    if rid in seen:
                        continue

                    rdoc = doc_lookup(rid)

                    if not rdoc:
                        continue

                    expanded.append(rdoc)
                    seen.add(rid)
                    next_frontier.append(rid)

                    count += 1

                    if len(expanded) >= max_expansion:
                        return candidates + expanded

                    if count >= self.per_doc_limit:
                        break

            frontier = next_frontier

            if not frontier:
                break

        return candidates + expanded