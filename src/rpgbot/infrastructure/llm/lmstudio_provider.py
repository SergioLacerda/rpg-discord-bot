from openai import AsyncOpenAI

from rpgbot.usecases.llm.llm_provider import LLMProvider
from rpgbot.core.providers import llm_registry


class LMStudioProvider(LLMProvider):

    def __init__(self, base_url: str, model: str, **_):

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="lm-studio"
        )

        self.model = model

    async def generate(self, prompt: str) -> str:

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        return resp.choices[0].message.content


@llm_registry.register("lmstudio")
def create_lmstudio_provider(**kwargs):
    return LMStudioProvider(**kwargs)