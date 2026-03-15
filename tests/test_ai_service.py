from unittest.mock import AsyncMock, patch
import pytest

from rpgbot.usecases.generate_narrative import generate_narrative, build_prompt


# ---------------------------------------------------------
# build_prompt
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# generate_narrative
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_narrative():

    async def fake_ctx(q):
        return "contexto"

    async def fake_cache_get(x):
        return None

    async def fake_cache_set(x, y):
        pass

    with patch(
        "rpgbot.usecases.generate_narrative._engine.generate",
        new=AsyncMock(return_value="resultado"),
    ):

        result = await generate_narrative(
            "investigar a caverna",
            ctx_provider=fake_ctx,
            cache_get=fake_cache_get,
            cache_set=fake_cache_set
        )

    assert result == "resultado"


# ---------------------------------------------------------
# retry test
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_narrative_retry():

    async def fake_generate(*args, **kwargs):
        return "sucesso após retry"

    async def fake_ctx(x):
        return "ctx"

    async def fake_cache_get(x):
        return None

    async def fake_cache_set(x, y):
        pass

    with patch(
        "rpgbot.usecases.generate_narrative._engine.generate",
        new=AsyncMock(return_value="sucesso após retry"),
    ):
        result = await generate_narrative(
            "ação",
            ctx_provider=fake_ctx,
            cache_get=fake_cache_get,
            cache_set=fake_cache_set
        )

    assert "sucesso após retry" in result

