import re
import hashlib


def embedding_key(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode()).hexdigest()


def compress_context(ctx: str, max_chars: int = 1200) -> str:

    if len(ctx) <= max_chars:
        return ctx

    head = ctx[: max_chars // 2]
    tail = ctx[-max_chars // 2 :]

    return f"{head}\n...\n{tail}"


def normalize_text(text: str) -> str:

    if not text:
        return ""

    return text.lower().strip()


def tokenize(text: str) -> list[str]:

    if not text:
        return []

    tokens = re.findall(r"\w+", text.lower())

    return [t for t in tokens if len(t) > 1]