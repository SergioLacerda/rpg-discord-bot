class FakeRetryLLMProvider:

    def __init__(self, fail_times=1):

        self.calls = 0
        self.fail_times = fail_times

    async def generate(self, prompt: str, **kwargs):

        self.calls += 1

        if self.calls <= self.fail_times:
            raise Exception("temporary error")

        return "sucesso após retry"
