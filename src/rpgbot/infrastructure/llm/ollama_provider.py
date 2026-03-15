import httpx

from rpgbot.usecases.llm.llm_provider import LLMProvider
from rpgbot.core.providers import llm_registry


class OllamaProvider(LLMProvider):

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        **_
    ):

        self.model = model
        self.base_url = base_url

    async def generate(self, prompt: str) -> str:

        async with httpx.AsyncClient() as client:

            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )

        data = response.json()

        text = data.get("response")

        if not text:
            raise RuntimeError("Ollama retornou resposta vazia")

        return text.strip()


@llm_registry.register(
    "ollama",
    aliases=["local"]
)
def create_ollama_provider(**kwargs):

    return OllamaProvider(**kwargs)