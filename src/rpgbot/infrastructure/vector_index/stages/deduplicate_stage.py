
class DeduplicateStage:

    def run(self, ctx, candidates):

        seen = set()
        unique = []

        for d in candidates:

            i = d["id"]

            if i not in seen:
                unique.append(d)
                seen.add(i)

        return unique