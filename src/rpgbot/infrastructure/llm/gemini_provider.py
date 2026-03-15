try:
    import google.generativeai as genai
except ImportError:
    genai = None

from rpgbot.usecases.llm.llm_provider import LLMProvider
from rpgbot.core.providers import llm_registry


if genai:

    class GeminiProvider(LLMProvider):

        def __init__(self, api_key: str, model: str, **_):

            genai.configure(api_key=api_key)

            self.model = genai.GenerativeModel(model)

        async def generate(self, prompt: str):

            resp = await self.model.generate_content_async(prompt)

            if not resp.text:
                raise RuntimeError("Gemini retornou resposta vazia")

            return resp.text.strip()


    @llm_registry.register("gemini", aliases=["google"])
    def create_gemini_provider(**kwargs):
        return GeminiProvider(**kwargs)