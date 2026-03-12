from datetime import datetime
from pathlib import Path
import asyncio
import aiofiles
import os

from rpgbot.core.config import settings

_log_lock = asyncio.Lock()


async def write_log(text: str) -> Path:

    settings.LOG_PATH.mkdir(parents=True, exist_ok=True)

    date = datetime.now().strftime("%Y-%m-%d")

    file = settings.LOG_PATH / f"session_{date}.md"
    tmp = file.with_suffix(".tmp")

    async with _log_lock:

        existing = ""
        if file.exists():
            async with aiofiles.open(file, "r", encoding="utf-8") as f:
                existing = await f.read()

        new_content = existing + text + "\n"

        async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
            await f.write(new_content)

        fd = os.open(tmp, os.O_RDWR)
        os.fsync(fd)
        os.close(fd)

        os.replace(tmp, file)

    return file