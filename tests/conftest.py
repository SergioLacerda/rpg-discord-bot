import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

from rpgbot.core.container import container
from rpgbot.bootstrap import setup_container


# ---------------------------------------------------------
# Load environment
# ---------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def load_test_env():

    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env.test"

    if env_file.exists():
        load_dotenv(env_file, override=True)

    os.environ.setdefault("RPG_ENV", "test")


# ---------------------------------------------------------
# DI bootstrap
# ---------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def setup_di():

    # garantir container limpo
    container.reset()

    setup_container()

    yield

    # opcional: limpar após testes
    container.reset()


# ---------------------------------------------------------
# Fake vector index
# ---------------------------------------------------------

@pytest.fixture
def fake_vector_index(mocker):

    index = mocker.Mock()

    async def fake_search(query, k=4):
        return ["fake context"]

    index.search = mocker.AsyncMock(side_effect=fake_search)

    # registrar no container
    container.register("vector_index", lambda: index)

    return index