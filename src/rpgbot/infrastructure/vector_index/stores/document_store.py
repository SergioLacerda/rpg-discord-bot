
class DocumentStore:

    def __init__(self):
        self.docs = {}

    def add(self, doc_id, text, source):
        self.docs[doc_id] = {
            "text": text,
            "source": source
        }

    def get(self, doc_id):
        return self.docs[doc_id]