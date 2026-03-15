from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, get_type_hints

from .env_loader import load_environment
from .settings import (
    Settings,
    RuntimeSettings,
    LLMSettings,
    EmbeddingSettings,
    AppSettings,
)



ENVIRONMENT, ENV_FILES, CLI_OVERRIDES = load_environment()


# ---------------------------------------------------------
# cached env access
# ---------------------------------------------------------

@lru_cache
def get_env(name: str, default: Any = None):

    return os.getenv(name, default)


def require_env(name: str):

    value = get_env(name)

    if value is None or value == "":
        raise RuntimeError(f"Missing required env variable: {name}")

    return value


# ---------------------------------------------------------
# type casting
# ---------------------------------------------------------

def cast(value: str, typ):

    if typ == int:
        return int(value)

    if typ == bool:
        return value.lower() in {"1", "true", "yes"}

    return value


def build_section(cls, mapping):

    hints = get_type_hints(cls)

    kwargs = {}

    for field, typ in hints.items():

        env_name = mapping.get(field)

        if not env_name:
            continue

        raw = get_env(env_name)

        if raw is None:
            kwargs[field] = None
            continue

        kwargs[field] = cast(raw, typ)

    return cls(**kwargs)


# ---------------------------------------------------------
# lazy settings loader
# ---------------------------------------------------------

@lru_cache
def load_settings():

    runtime = build_section(RuntimeSettings, {

        "environment": "ENVIRONMENT",
        "device": "DEVICE",

    })

    llm = build_section(LLMSettings, {

        "provider": "LLM_PROVIDER",
        "model": "LLM_MODEL",
        "api_key": "LLM_API_KEY",
        "base_url": "LLM_BASE_URL",

    })

    embeddings = build_section(EmbeddingSettings, {

        "provider": "EMBEDDING_PROVIDER",
        "model": "EMBEDDING_MODEL",
        "api_key": "EMBEDDING_API_KEY",
        "batch_size": "EMBEDDING_BATCH_SIZE",

    })

    app = AppSettings(

        discord_token=require_env("DISCORD_TOKEN"),
        max_cache_size=int(get_env("MAX_CACHE_SIZE", 10000))

    )

    return Settings(

        runtime=runtime,
        llm=llm,
        embeddings=embeddings,
        app=app
    )


settings = load_settings()


def debug_config():

    print("Environment:", ENVIRONMENT)
    print("Env files:", ENV_FILES)

    for k in sorted(os.environ):
        if k.startswith(("LLM_", "EMBEDDING_", "DISCORD_", "DEVICE")):
            print(f"{k}={os.environ[k]}")
