
class VectorStore:

    def __init__(self):
        self.vectors = {}

    def add(self, doc_id, vector):
        self.vectors[doc_id] = vector

    def get(self, doc_id):
        return self.vectors[doc_id]