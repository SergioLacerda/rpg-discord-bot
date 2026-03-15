import hashlib


# ---------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------

class MockLLMProvider:
    """
    Mock determinístico para testes.

    Pode retornar:
    - resposta fixa (tests)
    - resposta baseada em hash (default)
    """

    def __init__(self, fixed_response=None):

        self.fixed_response = fixed_response

    async def generate(self, prompt: str, **kwargs):

        if self.fixed_response:
            return self.fixed_response

        h = hashlib.sha1(prompt.encode()).hexdigest()[:8]

        return f"[mock:{h}] resposta simulada"


# ---------------------------------------------------------
# Mock Embeddings
# ---------------------------------------------------------

class MockEmbeddingProvider:

    async def embed(self, text):

        return [0.1] * 1536

    async def embed_batch(self, texts):

        return [[0.1] * 1536 for _ in texts]

