class NarrativeMemoryStage:

    priority = 5
    min_candidates = 0

    def __init__(self, memory_layers):

        self.memory_layers = memory_layers

    def run(self, ctx, candidates):

        results = []

        q_vec = ctx.q_vec

        for layer in self.memory_layers:

            docs = layer.search(q_vec)

            if docs:
                results.extend(docs)

            if len(results) >= 40:
                break

        if results:
            return results

        return candidates