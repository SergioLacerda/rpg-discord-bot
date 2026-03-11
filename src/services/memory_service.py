import time
import hashlib
import logging
from pathlib import Path

from src.infrastructure.embedding_cache import embed
from src.services.session_memory import hierarchical_search
from src.utils.json_store import load_json, save_json
from src.utils.vector_utils import cosine_similarity

logger = logging.getLogger(__name__)


CAMPAIGN_DIR = Path("campaign")
VECTOR_FILE = CAMPAIGN_DIR / "index_vectors.json"
NPC_FILE = CAMPAIGN_DIR / "npc_database.json"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunk_text(text, size=800):

    words = text.split()

    return [
        " ".join(words[i:i+size])
        for i in range(0, len(words), size)
    ]


def load_index():

    return load_json(VECTOR_FILE, [])


def search_context(query, k=4):

    docs = load_index()

    q_vec = embed(query)

    scored = sorted(
        ((cosine_similarity(q_vec, d["vector"]), d) for d in docs),
        reverse=True
    )

    return [d["text"] for _, d in scored[:k]]


def save_npc(name, description):

    db = load_json(NPC_FILE, {})

    db[name] = {
        "description": description,
        "last_seen": time.time()
    }

    save_json(NPC_FILE, db)


def get_npc(name):

    return load_json(NPC_FILE, {}).get(name)


def build_context(query):

    docs = search_context(query)

    memory = hierarchical_search(query)

    return "\n".join(docs + memory)

def index_campaign():

    existing = load_json(VECTOR_FILE, [])

    existing_map = {
        (doc["file"], doc.get("chunk", 0)): doc
        for doc in existing
    }

    updated_docs = []

    for file in CAMPAIGN_DIR.rglob("*.md"):

        text = file.read_text(encoding="utf-8")

        h = file_hash(file)

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):

            key = (str(file), i)

            if key in existing_map and existing_map[key]["hash"] == h:

                updated_docs.append(existing_map[key])
                continue

            vector = embed(chunk)

            updated_docs.append({
                "file": str(file),
                "hash": h,
                "chunk": i,
                "text": chunk,
                "vector": vector
            })

    save_json(VECTOR_FILE, updated_docs)

    return updated_docs