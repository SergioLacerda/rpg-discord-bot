from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

env_path = ROOT / ".env"

load_dotenv(env_path)

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não encontrado no .env")

LOG_PATH = ROOT / "logs"
LOG_PATH.mkdir(parents=True, exist_ok=True)