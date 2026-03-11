from src.infrastructure.embedding_client import embed


class FakeEmbedding:

    class Data:
        embedding = [1,2,3]

    data = [Data()]


class FakeClient:

    class Embeddings:

        def create(self, *args, **kwargs):
            return FakeEmbedding()

    embeddings = Embeddings()


def test_embed(monkeypatch):

    monkeypatch.setattr(
        "src.infrastructure.embedding_client.get_client",
        lambda: FakeClient()
    )

    vec = embed("texto")

    assert vec == [1,2,3]