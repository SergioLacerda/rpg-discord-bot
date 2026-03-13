import numpy as np
from pathlib import Path


class MemoryMappedVectorStore:

    def __init__(self, dim=768, path="campaign/vectors.dat"):

        self.dim = dim
        self.path = Path(path)

        self.id_to_pos = {}
        self.count = 0

        self._mmap = None

    def _ensure_mmap(self):

        if self._mmap is None:

            size = max(self.count, 1)

            self._mmap = np.memmap(
                self.path,
                dtype="float32",
                mode="w+",
                shape=(size, self.dim)
            )

    def add(self, doc_id, vector):

        self._ensure_mmap()

        pos = self.count

        self.id_to_pos[doc_id] = pos

        self._mmap[pos] = vector

        self.count += 1

    def get(self, doc_id):

        pos = self.id_to_pos.get(doc_id)

        if pos is None:
            return None

        return self._mmap[pos]

        def get_many(self, doc_ids):

            return [self.get(doc_id) for doc_id in doc_ids]