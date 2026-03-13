class ContextCompressor:

    def __init__(self, max_chars=1200):
        self.max_chars = max_chars

    def compress(self, history: str) -> str:

        if len(history) <= self.max_chars:
            return history

        parts = history.split("\n")

        trimmed = []
        size = 0

        for p in reversed(parts):

            size += len(p)

            if size > self.max_chars:
                break

            trimmed.append(p)

        return "\n".join(reversed(trimmed))