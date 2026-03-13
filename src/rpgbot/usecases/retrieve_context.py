from pathlib import Path

from rpgbot.core.container import container
from rpgbot.core.paths import CAMPAIGN_DIR


async def search_context(query, k=4, index=None):

    if index is None:
        index = container.resolve("vector_index")

    return await index.search(query, k=k)


async def hierarchical_context(query, k=4):
    return await search_context(query, k)


def get_index():
    return container.resolve("vector_index")


def get_campaign_index(campaign_id):

    index = container.resolve("vector_index")

    index.campaign_id = str(campaign_id)

    return index


async def index_campaign(campaign_dir: Path | None = None):

    campaign_dir = campaign_dir or CAMPAIGN_DIR

    index = get_index()

    campaign_files = list(campaign_dir.glob("**/*.md"))

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
            "mtime": file.stat().st_mtime
        })

    index.docs = docs

    return docs