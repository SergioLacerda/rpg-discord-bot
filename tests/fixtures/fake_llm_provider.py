
class FakeLLMProvider:

    async def generate(self, prompt):
        return "resultado"
