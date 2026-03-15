class NarrativeImportanceStage:

    priority = 45
    min_candidates = 1

    def __init__(self, importance_store):
        self.importance_store = importance_store

    def run(self, ctx, candidates):

        return sorted(
            candidates,
            key=lambda d: self.importance_store.get(d, 1),
            reverse=True
        )