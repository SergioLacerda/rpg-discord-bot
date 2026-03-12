import asyncio
from rpgbot.infrastructure.executor import EXECUTOR

async def run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EXECUTOR, func, *args)