from pathlib import Path

from .config import ROOT


# -----------------------------------------
# diretórios principais
# -----------------------------------------

CAMPAIGN_DIR = ROOT / "campaign"

MEMORY_DIR = CAMPAIGN_DIR / "memory"
LOG_DIR = ROOT / "logs"


# -----------------------------------------
# arquivos específicos
# -----------------------------------------

EMBEDDING_CACHE_PATH = MEMORY_DIR / "embedding_cache.json"

VECTOR_INDEX_FILE = CAMPAIGN_DIR / "vector.index"


# -----------------------------------------
# garantir estrutura
# -----------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_DIR.mkdir(parents=True, exist_ok=True)