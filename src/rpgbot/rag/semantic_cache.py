import hashlib


class HierarchicalSemanticCache:

    def __init__(self):

        self.query_cache = {}
        self.semantic_cache = {}

    def _normalize(self, text):

        return text.lower().strip()

    def query_key(self, query):

        q = self._normalize(query)
        return hashlib.sha1(q.encode()).hexdigest()

    def semantic_key(self, vec):

        return tuple(round(v, 2) for v in vec[:32])

    def get(self, query, vec):

        qk = self.query_key(query)

        if qk in self.query_cache:
            return self.query_cache[qk]

        sk = self.semantic_key(vec)

        if sk in self.semantic_cache:
            return self.semantic_cache[sk]

        return None

    def set(self, query, vec, result):

        self.query_cache[self.query_key(query)] = result
        self.semantic_cache[self.semantic_key(vec)] = result