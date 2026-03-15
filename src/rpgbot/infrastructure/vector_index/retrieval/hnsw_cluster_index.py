import hnswlib
import numpy as np


class HNSWClusterIndex:

    def __init__(self, dim, max_elements=10000):

        self.index = hnswlib.Index(space="cosine", dim=dim)

        self.index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16
        )

        self.doc_ids = []
        self.size = 0

    def add(self, doc_id, vector):

        self.index.add_items(
            np.array([vector]),
            [self.size]
        )

        self.doc_ids.append(doc_id)

        self.size += 1

    def search(self, vector, k=20):

        labels, distances = self.index.knn_query(
            vector,
            k=k
        )

        result = []

        for i in labels[0]:
            if i < len(self.doc_ids):
                result.append(self.doc_ids[i])

        return result