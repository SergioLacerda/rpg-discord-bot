
class VectorPrefilter:

    def run(self, ctx, candidates):

        scored = []

        for d in candidates:

            vec_score = cosine_similarity(ctx.q_vec, d["vector"])

            if vec_score > 0.05:
                scored.append((vec_score, d))

        scored.sort(reverse=True)

        return [d for _, d in scored[:80]]