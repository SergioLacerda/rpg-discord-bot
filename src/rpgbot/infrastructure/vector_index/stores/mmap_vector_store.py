import numpy as np
from pathlib import Path


class MMapVectorStore:

    def __init__(self, path: Path, dim: int, capacity: int = 1_000_000):

        self.path = path
        self.dim = dim
        self.capacity = capacity

        self.index = {}
        self.size = 0

        self.vectors = np.memmap(
            path,
            dtype="float32",
            mode="w+",
            shape=(capacity, dim)
        )

    def add(self, doc_id, vector):

        pos = self.size

        self.vectors[pos] = vector

        self.index[doc_id] = pos

        self.size += 1

    def get(self, doc_id):

        pos = self.index.get(doc_id)

        if pos is None:
            return None

        return self.vectors[pos]

    def clear(self):

        self.size = 0
        self.index.clear()