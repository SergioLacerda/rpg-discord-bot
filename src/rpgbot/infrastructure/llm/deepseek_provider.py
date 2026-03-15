try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from rpgbot.usecases.llm.llm_provider import LLMProvider
from rpgbot.core.providers import llm_registry


if AsyncOpenAI:

    class DeepSeekProvider(LLMProvider):

        def __init__(self, api_key: str, model: str, **_):

            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

            self.model = model

        async def generate(self, prompt: str):

            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )

            return resp.choices[0].message.content.strip()


    @llm_registry.register("deepseek")
    def create_deepseek_provider(**kwargs):
        return DeepSeekProvider(**kwargs)