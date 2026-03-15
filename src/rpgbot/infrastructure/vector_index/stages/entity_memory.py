class NarrativeEntityMemory:

    priority = 15
    min_candidates = 0

    def __init__(self, entity_memory):

        self.entity_memory = entity_memory

        self.alias_map = {}

        for name, data in entity_memory.items():

            for alias in data.get("aliases", []):
                self.alias_map[alias] = name

            self.alias_map[name] = name

    def run(self, ctx, candidates):

        tokens = ctx.query_tokens

        results = set(candidates)

        for t in tokens:

            ent = self.alias_map.get(t)

            if not ent:
                continue

            data = self.entity_memory.get(ent)

            if not data:
                continue

            docs = data.get("docs")

            if docs:
                results.update(docs)

        return list(results)