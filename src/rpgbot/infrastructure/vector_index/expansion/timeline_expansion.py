
class TimelineExpansion:

    priority = 48
    min_candidates = 1

    def __init__(self, window=1):

        self.window = window


    def run(self, ctx, candidates):

        index = ctx.index

        timeline = getattr(index, "timeline_index", None)

        if not timeline:
            return candidates

        doc_lookup = index.document_store.get

        expanded = []
        seen = {d["id"] for d in candidates}

        for doc in candidates:

            neighbors = timeline.neighbors(doc["id"], self.window)

            for nid in neighbors:

                if nid in seen:
                    continue

                ndoc = doc_lookup(nid)

                if ndoc:
                    expanded.append(ndoc)
                    seen.add(nid)

        return candidates + expanded