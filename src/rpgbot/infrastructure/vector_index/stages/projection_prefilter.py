
class ProjectionPrefilter:

    def run(self, ctx, candidates):

        q_proj = project(ctx.q_vec)

        threshold = max(0.15, 1 / (1 + len(ctx.docs) ** 0.5))

        filtered = [
            d for d in candidates
            if abs(d["proj"] - q_proj) < threshold
        ]

        return filtered if filtered else candidates