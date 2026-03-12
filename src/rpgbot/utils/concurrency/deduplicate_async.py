import asyncio
import time


class InflightDeduplicator:

    def __init__(self):

        self._inflight: dict[str, asyncio.Task] = {}
        self._recent: dict[str, tuple[float, object]] = {}

        self._ttl = 2.0  # segundos


    async def run(self, key: str, coro_factory):

        now = time.monotonic()

        # micro-cache recente
        if key in self._recent:
            ts, value = self._recent[key]

            if now - ts < self._ttl:
                return value

            del self._recent[key]

        # chamada já em andamento
        if key in self._inflight:
            return await self._inflight[key]

        task = asyncio.create_task(coro_factory())

        self._inflight[key] = task

        try:
            result = await task

            # guarda resultado recente
            self._recent[key] = (time.monotonic(), result)

            return result

        finally:
            self._inflight.pop(key, None)