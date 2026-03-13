import bisect
import time


class MetadataStore:
    """
    Armazena metadados de documentos com índice temporal.

    Features:
    - lookup O(1) por doc_id
    - busca temporal O(log n)
    - scoring de recência eficiente
    """

    def __init__(self):

        self.timestamps = {}
        self.mtimes = {}

        # índice temporal ordenado
        self.timeline = []
        self.timeline_docs = []

    # ---------------------------------------------------------
    # add metadata
    # ---------------------------------------------------------

    def add(self, doc_id, timestamp=None, mtime=None):

        ts = timestamp or mtime or 0

        self.timestamps[doc_id] = ts
        self.mtimes[doc_id] = mtime or ts

        # inserir no índice temporal
        pos = bisect.bisect_left(self.timeline, ts)

        self.timeline.insert(pos, ts)
        self.timeline_docs.insert(pos, doc_id)

    # ---------------------------------------------------------
    # basic getters
    # ---------------------------------------------------------

    def get(self, doc_id):

        return {
            "timestamp": self.timestamps.get(doc_id, 0),
            "mtime": self.mtimes.get(doc_id, 0)
        }

    def get_timestamp(self, doc_id):

        return self.timestamps.get(doc_id, 0)

    # ---------------------------------------------------------
    # recency score
    # ---------------------------------------------------------

    def recency_score(self, doc_id):

        """
        Score 0..1 baseado em quão recente é o documento.
        """

        ts = self.timestamps.get(doc_id)

        if not ts or not self.timeline:
            return 0.0

        pos = bisect.bisect_left(self.timeline, ts)

        # normalizar posição
        return pos / len(self.timeline)

    # ---------------------------------------------------------
    # recent documents
    # ---------------------------------------------------------

    def recent(self, k=10):

        """
        Retorna os k documentos mais recentes.
        """

        if not self.timeline_docs:
            return []

        return self.timeline_docs[-k:]

    # ---------------------------------------------------------
    # recent window (temporal queries)
    # ---------------------------------------------------------

    def recent_window(self, seconds=86400):

        """
        Retorna documentos dentro de uma janela temporal.
        Default: 24h
        """

        cutoff = time.time() - seconds

        pos = bisect.bisect_left(self.timeline, cutoff)

        return self.timeline_docs[pos:]

    # ---------------------------------------------------------
    # stats
    # ---------------------------------------------------------

    def size(self):

        return len(self.timestamps)