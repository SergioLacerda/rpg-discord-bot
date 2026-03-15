from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMSettings:

    provider: str
    model: str
    api_key: Optional[str]
    base_url: Optional[str]


@dataclass(frozen=True)
class EmbeddingSettings:

    provider: Optional[str]
    model: Optional[str]
    api_key: Optional[str]
    batch_size: Optional[int]


@dataclass(frozen=True)
class RuntimeSettings:

    environment: str
    device: Optional[str]


@dataclass(frozen=True)
class AppSettings:

    discord_token: str
    max_cache_size: int


@dataclass(frozen=True)
class Settings:

    runtime: RuntimeSettings
    llm: LLMSettings
    embeddings: EmbeddingSettings
    app: AppSettings
