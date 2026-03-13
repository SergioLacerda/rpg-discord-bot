class CausalExpansion:
    """
    Expande candidatos usando relações de causalidade entre documentos/eventos.
    """

    def __init__(self, max_expansion=20):
        self.max_expansion = max_expansion


    def run(self, ctx, candidates):

        index = ctx.index

        causality_graph = getattr(index, "causality_graph", None)

        if not causality_graph:
            return candidates

        doc_lookup = index.document_store.get

        expanded = []
        seen = {d["id"] for d in candidates}

        max_expansion = self.max_expansion

        for doc in candidates:

            doc_id = doc["id"]

            related_ids = causality_graph.get(doc_id)

            if not related_ids:
                continue

            for rid in related_ids:

                rdoc = doc_lookup(rid)

                if rdoc and rdoc["id"] not in seen:

                    expanded.append(rdoc)
                    seen.add(rdoc["id"])

                    if len(expanded) >= max_expansion:
                        return candidates + expanded

        return candidates + expanded