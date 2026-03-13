
class ANNPrefilter:

    def __init__(self, ann_index):
        self.ann_index = ann_index

    def run(self, ctx, candidates):

        if not self.ann_index:
            return candidates

        ann_candidates = self.ann_index.search(ctx.q_vec)

        if ann_candidates and len(ann_candidates) > 10:

            ann_set = {d["id"] for d in ann_candidates}

            candidates = [d for d in candidates if d["id"] in ann_set]

        return candidates