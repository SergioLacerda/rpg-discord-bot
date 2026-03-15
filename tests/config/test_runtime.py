import os
import socket

from rpgbot.bootstrap import setup_container
from rpgbot.core.providers import embedding_registry, llm_registry
from rpgbot.core.container import container

from rpgbot.infrastructure.narrative_memory import memory

from tests.fixtures.autoload import register_fake_providers
from tests.fixtures.mock_providers import MockEmbeddingProvider




class TestRuntime:
    """
    Inicializa ambiente de testes seguro e isolado.
    Também limpa qualquer estado global entre execuções.
    """

    _initialized = False

    @classmethod
    def bootstrap(cls):

        if cls._initialized:
            return

        # -----------------------------------------
        # environment
        # -----------------------------------------

        os.environ.setdefault("ENVIRONMENT", "test")

        # -----------------------------------------
        # container reset
        # -----------------------------------------

        container.reset()

        setup_container()

        # -----------------------------------------
        # fake providers
        # -----------------------------------------

        embedding_registry.register("mock", MockEmbeddingProvider)
        llm_registry.register("mock", MockLLMProvider)

        register_fake_providers(container)

        # -----------------------------------------
        # disable external network
        # -----------------------------------------

        socket.socket = cls._blocked_socket

        # -----------------------------------------
        # clean global states
        # -----------------------------------------

        cls._reset_memory()
        cls._reset_container_instances()
        cls._reset_vector_index()

        cls._initialized = True

    # -------------------------------------------------
    # network isolation
    # -------------------------------------------------

    @staticmethod
    def _blocked_socket(*args, **kwargs):

        raise RuntimeError(
            "External network disabled during tests"
        )

    # -------------------------------------------------
    # memory reset
    # -------------------------------------------------

    @staticmethod
    def _reset_memory():

        try:

            if hasattr(memory, "clear"):
                memory.clear()

            elif hasattr(memory, "events"):
                memory.events.clear()

            elif hasattr(memory, "history"):
                memory.history.clear()

        except Exception:
            pass

    # -------------------------------------------------
    # container instances
    # -------------------------------------------------

    @staticmethod
    def _reset_container_instances():

        try:
            container.reset_instances()
        except Exception:
            pass

    # -------------------------------------------------
    # vector index reset
    # -------------------------------------------------

    @staticmethod
    def _reset_vector_index():

        try:

            index = container.resolve("vector_index")

            if hasattr(index, "clear"):
                index.clear()

            if hasattr(index, "doc_ids"):
                index.doc_ids.clear()

            if hasattr(index, "vector_store") and hasattr(index.vector_store, "clear"):
                index.vector_store.clear()

        except Exception:
            pass


    def disable_network():

        real_socket = socket.socket

        def guard(*args, **kwargs):
            raise RuntimeError("External network disabled during tests")

        socket.socket = guard
