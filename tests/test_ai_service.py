import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rpgbot.services.ai_service import generate_narrative, build_prompt


class FakeMessage:
    content = "Narrativa teste"


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]

@pytest.mark.asyncio
async def test_prompt_contains_player_action(monkeypatch):
    monkeypatch.setattr(
        "rpgbot.services.ai_service.hierarchical_context",
        AsyncMock(return_value="Contexto falso aqui")
    )

    prompt = await build_prompt("entro no armazém")

    assert "entro no armazém" in prompt
    assert "Contexto falso aqui" in prompt


@pytest.mark.asyncio
async def test_generate_narrative(monkeypatch):
    fake_response = FakeResponse()
    fake_response.choices[0].message.content = "Narrativa teste mockada"

    fake_create = AsyncMock(return_value=fake_response)

    fake_completions = MagicMock()
    fake_completions.create = fake_create

    fake_chat = MagicMock()
    fake_chat.completions = fake_completions

    fake_client = MagicMock()
    fake_client.chat = fake_chat

    with patch("rpgbot.services.ai_service.get_client") as mock_get_client:
        mock_get_client.return_value = fake_client

        monkeypatch.setattr(
            "rpgbot.services.ai_service.hierarchical_context",
            AsyncMock(return_value="contexto fake")
        )

        result = await generate_narrative("entro no galpão")

    assert result == "Narrativa teste mockada"
    assert fake_create.call_count == 1


@pytest.mark.asyncio
async def test_generate_narrative_retry(monkeypatch):
    calls = {"n": 0}

    async def fake_create_side_effect(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RateLimitError(
                message=f"Rate limit simulado tentativa {calls['n']}",
                response=MagicMock(status_code=429),
                body={"error": {"message": "rate limit"}},
                code=429
            )
        response = FakeResponse()
        response.choices[0].message.content = "Narrativa gerada com sucesso após retry"
        return response

    fake_create = AsyncMock(side_effect=fake_create_side_effect)

    fake_completions = MagicMock()
    fake_completions.create = fake_create

    fake_chat = MagicMock()
    fake_chat.completions = fake_completions

    fake_client = MagicMock()
    fake_client.chat = fake_chat

    # Patch no namespace do ai_service
    with patch("rpgbot.services.ai_service.get_client") as mock_get_client:
        mock_get_client.return_value = fake_client

        monkeypatch.setattr(
            "rpgbot.services.ai_service.hierarchical_context",
            AsyncMock(return_value="ctx para retry")
        )

        result = await generate_narrative("ação de teste para retry")

    assert "sucesso após retry" in result
    assert calls["n"] == 2
    assert fake_create.call_count == 2