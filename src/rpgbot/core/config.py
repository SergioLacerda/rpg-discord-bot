from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv


# -----------------------------------------
# encontrar raiz do projeto automaticamente
# -----------------------------------------

def find_project_root(start: Path) -> Path:

    current = start.resolve()

    for parent in [current] + list(current.parents):

        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError("Não foi possível encontrar a raiz do projeto")


ROOT = find_project_root(Path(__file__))


# -----------------------------------------
# selecionar ambiente
# -----------------------------------------

ENVIRONMENT = os.getenv("RPG_ENV", "dev")

ENV_FILES = [
    ROOT / ".env",
    ROOT / f".env.{ENVIRONMENT}",
]


# -----------------------------------------
# carregar dotenv
# -----------------------------------------

for env_file in ENV_FILES:

    if env_file.exists():
        load_dotenv(env_file, override=True)


# -----------------------------------------
# utilitário de leitura
# -----------------------------------------

def require_env(name: str) -> str:

    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"{name} não encontrado no ambiente")

    return value


# -----------------------------------------
# settings
# -----------------------------------------

@dataclass(frozen=True)
class Settings:

    DISCORD_TOKEN: str
    OPENAI_API_KEY: str

    MAX_CACHE_SIZE: int
    EMBEDDING_CACHE_PATH: Path
    LOG_PATH: Path


settings = Settings(

    DISCORD_TOKEN=require_env("DISCORD_TOKEN"),
    OPENAI_API_KEY=require_env("OPENAI_API_KEY"),

    MAX_CACHE_SIZE=int(os.getenv("MAX_CACHE_SIZE", "10000")),

    EMBEDDING_CACHE_PATH=ROOT / "campaign" / "memory" / "embedding_cache.json",

    LOG_PATH=ROOT / "logs",
)


settings.LOG_PATH.mkdir(parents=True, exist_ok=True)