import pytest
from rpgbot.adapters.storage.file_log_repository import write_log

@pytest.mark.asyncio
async def test_log_write():

    file = await write_log("teste")

    assert file.exists()