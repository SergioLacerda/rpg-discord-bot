import heapq


class TopK:

    def __init__(self, k):

        self.k = k
        self.heap = []

    def push(self, score, item):

        if len(self.heap) < self.k:

            heapq.heappush(self.heap, (score, item))

        else:

            heapq.heappushpop(self.heap, (score, item))

    def results(self):

        return [
            item for _, item in sorted(self.heap, reverse=True)
        ]