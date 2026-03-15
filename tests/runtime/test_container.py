from rpgbot.core.container import container


def register_fake_llm(result="ok"):

    class FakeLLMProvider:

        async def generate(self, prompt, **kwargs):
            return result

    container.register(
        "llm_provider",
        lambda: FakeLLMProvider(),
        singleton=True
    )
