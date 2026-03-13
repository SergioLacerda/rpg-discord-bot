
class EntityRecallRouter:

    def __init__(self, entity_resolver=None, scan_limit=200):

        self.entity_resolver = entity_resolver
        self.scan_limit = scan_limit


    def run(self, ctx, candidates):

        docs = ctx.index.docs

        if not self.entity_resolver:
            return candidates

        related = getattr(ctx, "entities", None)

        if related is None:
            related = {e.lower() for e in self.entity_resolver(ctx.query)}
            ctx.entities = related

        if not related:
            return candidates

        seen = {d["id"] for d in candidates}

        for doc in docs[: self.scan_limit]:

            if doc["token_set"] & related and doc["id"] not in seen:

                candidates.append(doc)
                seen.add(doc["id"])

        return candidates