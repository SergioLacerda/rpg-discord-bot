import bisect
import time
from collections import defaultdict


class TemporalMemoryIndex:
    """
    Índice temporal para eventos narrativos.

    Features:
    - timeline ordenada
    - recency scoring O(log n)
    - detecção simples de sequências de eventos
    - busca por janela temporal
    """

    def __init__(self):

        # metadata base
        self.timestamps = {}

        # timeline ordenada
        self.timeline = []
        self.timeline_docs = []

        # relações causais simples
        self.event_graph = defaultdict(list)

        # tokens/eventos associados
        self.event_tokens = {}

    # ---------------------------------------------------------
    # add event
    # ---------------------------------------------------------

    def add(self, doc_id, timestamp=None, tokens=None):

        ts = timestamp or time.time()

        self.timestamps[doc_id] = ts
        self.event_tokens[doc_id] = set(tokens or [])

        pos = bisect.bisect_left(self.timeline, ts)

        self.timeline.insert(pos, ts)
        self.timeline_docs.insert(pos, doc_id)

    # ---------------------------------------------------------
    # recency score
    # ---------------------------------------------------------

    def recency_score(self, doc_id):

        ts = self.timestamps.get(doc_id)

        if ts is None or not self.timeline:
            return 0.0

        pos = bisect.bisect_left(self.timeline, ts)

        return pos / len(self.timeline)

    # ---------------------------------------------------------
    # recent events
    # ---------------------------------------------------------

    def recent(self, k=10):

        return self.timeline_docs[-k:]

    # ---------------------------------------------------------
    # window search
    # ---------------------------------------------------------

    def window(self, seconds):

        cutoff = time.time() - seconds

        pos = bisect.bisect_left(self.timeline, cutoff)

        return self.timeline_docs[pos:]

    # ---------------------------------------------------------
    # build causal relations
    # ---------------------------------------------------------

    def build_sequences(self, window=5):

        """
        Detecta sequências de eventos baseadas em proximidade temporal
        e tokens compartilhados.
        """

        for i, doc_id in enumerate(self.timeline_docs):

            tokens_i = self.event_tokens.get(doc_id, set())

            for j in range(1, window):

                if i + j >= len(self.timeline_docs):
                    break

                next_doc = self.timeline_docs[i + j]

                tokens_j = self.event_tokens.get(next_doc, set())

                # relação simples baseada em interseção semântica
                if tokens_i & tokens_j:

                    self.event_graph[doc_id].append(next_doc)

    # ---------------------------------------------------------
    # causal expansion
    # ---------------------------------------------------------

    def causal_chain(self, doc_id, depth=3):

        """
        Retorna cadeia de eventos relacionados.
        """

        result = []
        frontier = [doc_id]

        for _ in range(depth):

            new_frontier = []

            for d in frontier:

                children = self.event_graph.get(d, [])

                result.extend(children)

                new_frontier.extend(children)

            frontier = new_frontier

        return result

    # ---------------------------------------------------------
    # query helper
    # ---------------------------------------------------------

    def investigate(self, tokens, k=10):

        """
        Busca eventos que compartilham tokens com a query.
        """

        tokens = set(tokens)

        results = []

        for doc_id, doc_tokens in self.event_tokens.items():

            if tokens & doc_tokens:

                results.append(doc_id)

        results.sort(key=lambda d: self.timestamps.get(d, 0), reverse=True)

        return results[:k]

    # ---------------------------------------------------------
    # stats
    # ---------------------------------------------------------

    def size(self):

        return len(self.timestamps)