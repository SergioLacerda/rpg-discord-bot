from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def find_project_root(start: Path) -> Path:

    current = start.resolve()

    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError("Project root not found")


ROOT = find_project_root(Path(__file__).parent)


def parse_cli_overrides():

    overrides = {}

    for arg in sys.argv:

        if not arg.startswith("--"):
            continue

        if "=" not in arg:
            continue

        key, value = arg[2:].split("=", 1)

        overrides[key.upper()] = value

    return overrides


def load_environment():

    env = os.getenv("ENVIRONMENT", "dev").lower()

    env_files = [
        ROOT / ".env",
        ROOT / f".env.{env}",
        ROOT / ".env.local",
        ROOT / f".env.{env}.local",
    ]

    loaded = []

    for file in env_files:

        if file.exists():
            load_dotenv(file, override=False)
            loaded.append(file.name)

    cli_overrides = parse_cli_overrides()

    for k, v in cli_overrides.items():
        os.environ[k] = v

    return env, loaded, cli_overrides
