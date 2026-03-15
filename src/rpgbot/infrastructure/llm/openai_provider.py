from openai import AsyncOpenAI

from rpgbot.usecases.llm.llm_provider import LLMProvider
from rpgbot.core.providers import llm_registry


class OpenAIProvider(LLMProvider):

    def __init__(self, api_key: str, model: str, base_url=None):

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.model = model

    async def generate(self, prompt: str) -> str:

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        return resp.choices[0].message.content


@llm_registry.register("openai")
def create_openai_provider(**kwargs):
    return OpenAIProvider(**kwargs)