class VectorANNIndex:

    def __init__(self, docs, bucket_count=32):

        self.bucket_count = bucket_count
        self.buckets = {i: [] for i in range(bucket_count)}

        for d in docs:

            h = hash(tuple(d["vector"][:8])) % bucket_count

            self.buckets[h].append(d)

    def search(self, query_vec):

        h = hash(tuple(query_vec[:8])) % self.bucket_count

        return self.buckets.get(h, [])