
from rpgbot.utils import load_json, save_json


class DocumentRepository:

    def __init__(self, vector_file):
        self.vector_file = vector_file

    def load(self):
        return load_json(self.vector_file, [])

    def save(self, docs):
        save_json(self.vector_file, docs)