from rpgbot.bootstrap import setup_container
from rpgbot.core.container import container

def test_vector_index_bootstrap():

    setup_container()

    index = container.resolve("vector_index")

    assert index is not None