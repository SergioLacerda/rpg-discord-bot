class NarrativeWindowRetriever:

    priority = 35
    min_candidates = 1

    def __init__(self, docs, window_before=2, window_after=2):

        self.docs = docs
        self.window_before = window_before
        self.window_after = window_after

        # mapa rápido doc_id -> posição
        self.positions = {doc_id: i for i, doc_id in enumerate(docs)}

    def run(self, ctx, candidates):

        if not candidates:
            return candidates

        expanded = set(candidates)

        for doc_id in candidates:

            pos = self.positions.get(doc_id)

            if pos is None:
                continue

            start = max(0, pos - self.window_before)
            end = min(len(self.docs), pos + self.window_after + 1)

            for i in range(start, end):
                expanded.add(self.docs[i])

        return list(expanded)