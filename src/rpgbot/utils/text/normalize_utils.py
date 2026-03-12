import hashlib


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def embedding_key(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode()).hexdigest()