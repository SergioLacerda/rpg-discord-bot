import asyncio
import inspect


async def resilient_call(
    providers,
    *args,
    retries=3,
    backoff=1.5,
    speculative_delay=None,
    **kwargs
):

    async def call(fn):

        delay = 0.1

        for attempt in range(retries):

            try:

                result = fn(*args, **kwargs)

                if inspect.isawaitable(result):
                    result = await result

                return result

            except Exception:

                if attempt >= retries - 1:
                    raise

                await asyncio.sleep(delay)
                delay *= backoff

    # ---------------------------------------------------------
    # speculative execution
    # ---------------------------------------------------------

    if speculative_delay and len(providers) > 1:

        primary = providers[0]
        fallback = providers[1]

        primary_task = asyncio.create_task(call(primary))

        await asyncio.sleep(speculative_delay)

        if primary_task.done():
            return await primary_task

        fallback_task = asyncio.create_task(call(fallback))

        done, pending = await asyncio.wait(
            [primary_task, fallback_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()

        return list(done)[0].result()

    # ---------------------------------------------------------
    # normal execution
    # ---------------------------------------------------------

    last_error = None

    for provider in providers:

        try:
            return await call(provider)

        except Exception as e:
            last_error = e

    raise last_error