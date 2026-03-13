
class IVFIndex:

    def __init__(self):
        self.centroids = []
        self.inverted_lists = {}
        self.doc_to_cluster = {}

    def add(self, doc_id, vector):

        best = None
        best_score = -1

        for i, c in enumerate(self.centroids):

            s = cosine_similarity(vector, c)

            if s > best_score:
                best = i
                best_score = s

        self.inverted_lists.setdefault(best, []).append(doc_id)
        self.doc_to_cluster[doc_id] = best