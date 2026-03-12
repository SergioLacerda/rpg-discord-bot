from unittest.mock import MagicMock
import pytest

from openai import RateLimitError

from rpgbot.usecases.generate_narrative import generate_narrative, build_prompt


class FakeMessage:
    content = "Narrativa teste"


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]


# -----------------------------
# TESTE build_prompt
# -----------------------------

@pytest.mark.asyncio
async def test_prompt_contains_player_action():

    async def fake_context(q):
        return "Contexto falso"

    prompt = await build_prompt(
        "abrir a porta",
        ctx_provider=fake_context
    )

    assert "abrir a porta" in prompt
    assert "Contexto falso" in prompt


# -----------------------------
# TESTE generate_narrative
# -----------------------------

@pytest.mark.asyncio
async def test_generate_narrative():

    async def fake_ctx(q):
        return "contexto"

    async def fake_cache_get(x):
        return None

    async def fake_cache_set(x, y):
        pass

    class FakeClient:

        class Chat:
            class Completions:

                async def create(self, *args, **kwargs):

                    class Response:
                        class Choice:
                            class Message:
                                content = "resultado"
                            message = Message()

                        choices = [Choice()]

                    return Response()

            completions = Completions()

        chat = Chat()

    result = await generate_narrative(
        "investigar a caverna",
        client=FakeClient(),
        ctx_provider=fake_ctx,
        cache_get=fake_cache_get,
        cache_set=fake_cache_set
    )

    assert result == "resultado"


# -----------------------------
# TESTE retry (RateLimit)
# -----------------------------

@pytest.mark.asyncio
async def test_generate_narrative_retry():

    calls = {"n": 0}

    async def fake_create(*args, **kwargs):

        calls["n"] += 1

        if calls["n"] < 2:
            raise RateLimitError(
                "rate limit",
                response=MagicMock(),
                body={}
            )

        class FakeResponse:
            class Choice:
                class Message:
                    content = "sucesso após retry"
                message = Message()

            choices = [Choice()]

        return FakeResponse()

    class FakeClient:
        class Chat:
            class Completions:
                create = fake_create

            completions = Completions()

        chat = Chat()

    async def fake_ctx(x):
        return "ctx"

    async def fake_cache_get(x):
        return None

    async def fake_cache_set(x, y):
        pass

    result = await generate_narrative(
        "ação",
        client=FakeClient(),
        ctx_provider=fake_ctx,
        cache_get=fake_cache_get,
        cache_set=fake_cache_set
    )

    assert "sucesso após retry" in result
    assert calls["n"] == 2