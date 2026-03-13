import os
from pathlib import Path

from rpgbot.core.container import container
from rpgbot.core.providers import lazy
from rpgbot.usecases.retrieval_engine import RetrievalEngine

CAMPAIGN_DIR = Path("campaign")

_engine = lazy("retrieval_engine")


async def search_context(query, k=4, index=None):

    if index is None:
        index = container.resolve("vector_index")

    return await index.search(query, k=k)


async def hierarchical_context(query, k=4):
    return await search_context(query, k)


def get_index():

    index = container.resolve("vector_index")

    if hasattr(index, "campaign_dir"):
        index.campaign_dir = CAMPAIGN_DIR

    return index


def get_campaign_index(campaign_id):

    index = container.resolve("vector_index")

    index.campaign_id = str(campaign_id)

    return index


async def index_campaign():

    index = get_index()

    index.campaign_dir = CAMPAIGN_DIR

    # arquivos markdown no diretório da campanha
    campaign_files = list(CAMPAIGN_DIR.glob("**/*.md"))

    if not campaign_files:
        index.docs = []
        return []

    docs = []

    embed = container.resolve("embed")

    for file in campaign_files:

        text = file.read_text(encoding="utf-8")

        vec = await embed(text)

        docs.append({
            "text": text,
            "vector": vec,
            "source": str(file),
            "mtime": os.path.getmtime(file)
        })

    index.docs = docs

    return docs