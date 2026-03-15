try:
    import anthropic
except ImportError:
    anthropic = None

from rpgbot.usecases.llm.llm_provider import LLMProvider
from rpgbot.core.providers import llm_registry


if anthropic:

    class AnthropicProvider(LLMProvider):

        def __init__(self, api_key: str, model: str, **_):

            if anthropic is None:
                raise RuntimeError(
                    "Anthropic provider requires 'anthropic' package. "
                    "Install with: pip install anthropic"
                )

            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            self.model = model

            async def generate(self, prompt: str):

                resp = await self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )

                return resp.content[0].text

        
        @llm_registry.register("anthropic")
        def create_anthropic_provider(**kwargs):
            return AnthropicProvider(**kwargs)