# src/infrastructure/vector_store.py
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional

from rpgbot.infrastructure.embedding_cache import embed
from rpgbot.infrastructure.embedding_client import remote_embed

CHROMA_PATH = Path("./chroma_db").resolve()
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None

def get_chroma_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
    return _client


def get_collection(name: str = "rpg_memory") -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_chroma_client()
        try:
            _collection = client.get_collection(name=name)
        except ValueError:
            _collection = client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}  # métrica mais comum para embeddings OpenAI
            )
    return _collectiontest_generate_narrative