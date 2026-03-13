
class QueryClassifier:

    def __init__(self):

        self.lore_patterns = (
            "quem é",
            "quem foi",
            "o que é",
            "quem são",
            "who is",
            "what is",
        )

        self.memory_patterns = (
            "o que aconteceu",
            "o que houve",
            "quando aconteceu",
            "recentemente",
            "what happened",
            "recent",
        )

        self.investigation_patterns = (
            "por que",
            "quem causou",
            "como aconteceu",
            "o que levou",
            "why",
            "what caused",
        )

    # ---------------------------------------------------------
    # classify query
    # ---------------------------------------------------------

    def classify(self, query: str) -> str:

        q = query.lower()

        if any(p in q for p in self.investigation_patterns):
            return "investigation"

        if any(p in q for p in self.lore_patterns):
            return "lore"

        if any(p in q for p in self.memory_patterns):
            return "memory"

        return "semantic"