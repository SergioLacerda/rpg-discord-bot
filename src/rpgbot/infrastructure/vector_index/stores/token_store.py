
class TokenStore:

    def __init__(self):
        self.tokens = {}

    def add(self, doc_id, tokens, token_set):
        self.tokens[doc_id] = (tokens, token_set)

    def get(self, doc_id):
        return self.tokens[doc_id]