from __future__ import annotations

import os
import pkgutil
import importlib
import inspect
import pytest
from pathlib import Path
from dotenv import load_dotenv

from rpgbot.core.container import container
from rpgbot.core.providers import embedding_registry

from tests.fixtures.embedding_reset import reset_embedding_state
from tests.runtime.test_container import register_fake_llm


# --------------------------------------------------
# project root
# --------------------------------------------------

def find_project_root(start: Path) -> Path:

    current = start.resolve()

    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError("Project root not found")


ROOT = find_project_root(Path(__file__).parent)


# --------------------------------------------------
# dotenv loading (TEST MODE)
# --------------------------------------------------

def load_test_environment():

    env_file = ROOT / ".env.test"

    if env_file.exists():
        load_dotenv(env_file, override=True)

    os.environ.setdefault("ENVIRONMENT", "test")


load_test_environment()


# --------------------------------------------------
# register mock embedding provider
# --------------------------------------------------

class MockEmbeddingProvider:

    async def embed(self, text):
        return [0.0] * 1536

    async def embed_batch(self, texts):
        return [[0.0] * 1536 for _ in texts]


# evita registrar duas vezes
if "mock" not in embedding_registry._providers:

    @embedding_registry.register("mock")
    def _mock_embedding_provider(**kwargs):
        return MockEmbeddingProvider()


# --------------------------------------------------
# Auto Mock Loader
# --------------------------------------------------

FIXTURE_PACKAGE = "tests.fixtures"


def discover_fake_providers():

    providers = {}

    try:
        pkg = importlib.import_module(FIXTURE_PACKAGE)
    except Exception:
        return providers

    for module_info in pkgutil.iter_modules(pkg.__path__):

        module = importlib.import_module(
            f"{FIXTURE_PACKAGE}.{module_info.name}"
        )

        for name, obj in inspect.getmembers(module, inspect.isclass):

            if name.startswith("Fake") and name.endswith("Provider"):

                service = name[4:-8].lower()

                providers[f"{service}_provider"] = obj

    return providers


# --------------------------------------------------
# Test Container Sandbox
# --------------------------------------------------

@pytest.fixture(autouse=True)
def test_container():

    container.reset()

    # --------------------------------------------------
    # auto mock providers
    # --------------------------------------------------

    providers = discover_fake_providers()

    for name, cls in providers.items():

        # evita sobrescrever providers explícitos
        if name not in container._providers:

            container.register(
                name,
                lambda cls=cls: cls(),
                singleton=True
            )

    # --------------------------------------------------
    # Fake Embedding Provider
    # --------------------------------------------------

    class FakeEmbeddingProvider:

        async def embed(self, text):
            return [1.0, 0.0, 0.0]

        async def embed_batch(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    container.register(
        "embedding_provider",
        lambda: FakeEmbeddingProvider(),
        singleton=True
    )

    container.register(
        "embed",
        lambda embedding_provider: embedding_provider.embed,
        singleton=True
    )

    # --------------------------------------------------
    # Fake Vector Index
    # --------------------------------------------------

    class FakeVectorIndex:

        async def search(self, query, k=4):
            return ["Stormy infiltrou o armazém"]

    container.register(
        "vector_index",
        lambda: FakeVectorIndex(),
        singleton=True
    )

    # --------------------------------------------------
    # Fake LLM Provider padrão
    # --------------------------------------------------

    register_fake_llm("resultado")

    yield container

    container.reset()


# --------------------------------------------------
# reset embedding system
# --------------------------------------------------

@pytest.fixture(autouse=True)
def reset_embeddings():

    reset_embedding_state()

    yield

    reset_embedding_state()


# --------------------------------------------------
# override LLM provider
# --------------------------------------------------

@pytest.fixture
def fake_llm():

    register_fake_llm()

    yield
