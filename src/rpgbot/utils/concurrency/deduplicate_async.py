import asyncio
import weakref


class InflightDeduplicator:

    def __init__(self):
        self._lock = asyncio.Lock()
        self._futures = weakref.WeakValueDictionary()

    async def run(self, key, coro):

        async with self._lock:

            future = self._futures.get(key)

            if future is None:

                loop = asyncio.get_running_loop()
                future = loop.create_future()

                self._futures[key] = future
                creator = True

            else:
                creator = False

        if not creator:
            return await future

        try:

            result = await coro()

            future.set_result(result)

            return result

        except Exception as e:

            future.set_exception(e)
            raise

        finally:

            async with self._lock:
                self._futures.pop(key, None)