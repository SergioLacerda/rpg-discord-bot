import time
from collections import OrderedDict


class NarrativeLRUCache:
    """
    Cache LRU com TTL para respostas narrativas.

    Usado antes da chamada ao NarrativeEngine para evitar
    execuções repetidas de LLM.

    Características:
    - LRU eviction
    - TTL automático
    - thread-safe para asyncio
    """

    def __init__(self, max_size=512, ttl=600):

        self.max_size = max_size
        self.ttl = ttl

        self.store = OrderedDict()

    # ---------------------------------------------------------
    # get
    # ---------------------------------------------------------

    async def get(self, key):

        item = self.store.get(key)

        if not item:
            return None

        value, expires = item

        # TTL expired
        if expires and time.time() > expires:

            self.store.pop(key, None)

            return None

        # move to end (LRU)
        self.store.move_to_end(key)

        return value

    # ---------------------------------------------------------
    # set
    # ---------------------------------------------------------

    async def set(self, key, value):

        expires = time.time() + self.ttl if self.ttl else None

        self.store[key] = (value, expires)

        self.store.move_to_end(key)

        # LRU eviction
        if len(self.store) > self.max_size:

            self.store.popitem(last=False)

    # ---------------------------------------------------------
    # cleanup expired
    # ---------------------------------------------------------

    def cleanup(self):

        now = time.time()

        expired = [
            k for k, (_, exp) in self.store.items()
            if exp and now > exp
        ]

        for k in expired:
            self.store.pop(k, None)