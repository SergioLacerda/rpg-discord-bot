import inspect
from rpgbot.utils.hash_utils import sha256_hash

class EmbeddingIndexer:

    def __init__(self, embed, feature_store):
        self.embed = embed
        self.feature_store = feature_store

    # ---------------------------------------------------------
    # embed helper
    # ---------------------------------------------------------

    async def _embed(self, text):

        if inspect.iscoroutinefunction(self.embed):
            return await self.embed(text)

        return self.embed(text)

    # ---------------------------------------------------------
    # build embeddings
    # ---------------------------------------------------------

    async def build(self, raw_docs, persisted):

        existing = {
            d["source"]: d
            for d in persisted
            if d.get("source")
        }

        updated_docs = []

        for doc in raw_docs:

            source = doc["source"]
            text = doc["text"]
            mtime = doc["mtime"]

            text_hash = sha256_hash(text[:4096])

            prev = existing.get(source)

            if prev and prev.get("hash") == text_hash:
                updated_docs.append(prev)
                continue

            vec = await self._embed(text)

            updated_docs.append({
                "id": source,
                "text": text,
                "vector": vec,
                "source": source,
                "mtime": mtime,
                "hash": text_hash,
                "timestamp": mtime,
                "tokens": doc.get("tokens", []),
                "token_set": doc.get("token_set", set())
            })

        entities = set(tokens)

        context_score = len(tokens)

        self.feature_store.add(
            source,
            entities,
            context_score
        )

        return updated_docs