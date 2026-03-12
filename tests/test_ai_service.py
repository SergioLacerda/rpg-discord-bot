from rpgbot.services.ai_service import generate_narrative
from rpgbot.services.ai_service import build_prompt


class FakeResponse:

    class Choice:
        class Message:
            content = "Narrativa teste"

        message = Message()

    choices = [Choice()]


class FakeClient:

    class Chat:
        class Completions:

            def create(self, *args, **kwargs):
                return FakeResponse()

        completions = Completions()

    chat = Chat()


def test_prompt_contains_player_action(monkeypatch):

    monkeypatch.setattr(
        "rpgbot.services.ai_service.hierarchical_context",
        lambda q: "Contexto falso"
    )

    prompt = build_prompt("entro no armazém")

    assert "entro no armazém" in prompt


def test_generate_narrative(monkeypatch):

    monkeypatch.setattr(
        "rpgbot.services.ai_service.get_client",
        lambda: FakeClient()
    )

    monkeypatch.setattr(
        "rpgbot.services.ai_service.hierarchical_context",
        lambda q: "contexto fake"
    )

    result = generate_narrative("entro no galpão")

    assert result == "Narrativa teste"

def test_generate_narrative_retry(monkeypatch):

    calls = {"n":0}

    class FakeClient:

        class Chat:
            class Completions:

                def create(self, *args, **kwargs):

                    calls["n"] += 1

                    if calls["n"] < 2:
                        raise Exception("fail")

                    class Response:
                        class Choice:
                            class Message:
                                content = "ok"

                            message = Message()

                        choices = [Choice()]

                    return Response()

            completions = Completions()

        chat = Chat()

    monkeypatch.setattr(
        "rpgbot.services.ai_service.get_client",
        lambda: FakeClient()
    )

    monkeypatch.setattr(
        "rpgbot.services.ai_service.hierarchical_context",
        lambda x: "ctx"
    )

    result = generate_narrative("teste")

    assert result == "ok"