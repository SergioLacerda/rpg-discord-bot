
def register_fake_llm(result="ok"):
    class FakeLLM:
        async def generate(self, *a, **k):
            return result

    container.register("llm_provider", lambda: FakeLLM(), singleton=True)
