import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Callable, Awaitable, Optional

from rpgbot.utils import load_json, save_json
from rpgbot.utils.text.normalize_utils import tokenize
from rpgbot.campaign.memory.entity_alias_resolver import EntityAliasResolver
from rpgbot.core.runtime_state import bump_event_version
from rpgbot.infrastructure.narrative_graph import update_graph_from_event

logger = logging.getLogger(__name__)


class AsyncJSONRepository:

    def __init__(self, flush_interval: float = 0.5):

        self.events_file = Path("campaign/memory/events.json")
        self.sessions_file = Path("campaign/memory/sessions.json")
        self.arcs_file = Path("campaign/memory/arcs.json")

        self._cache: dict[str, list] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._dirty: set[str] = set()

        self._flush_interval = flush_interval
        self._flush_task: Optional[asyncio.Task] = None

        self.alias_resolver = EntityAliasResolver()

        self.MAX_EVENTS = 200
        self.MAX_SESSIONS = 100
        self.MAX_EVENT_CHARS = 3000

    # ---------------------------------------------------------
    # utils
    # ---------------------------------------------------------

    def _key(self, path: Path) -> str:
        return str(path.resolve())

    def _get_lock(self, path: Path) -> asyncio.Lock:

        key = self._key(path)

        lock = self._locks.get(key)

        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock

        return lock

    def _ensure_file(self, path: Path):

        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            save_json(path, [])

    # ---------------------------------------------------------
    # load
    # ---------------------------------------------------------

    async def load(self, path: Path):

        key = self._key(path)

        if key in self._cache:
            return self._cache[key]

        self._ensure_file(path)

        try:

            data = load_json(path, [])

            self._cache[key] = data

            return data

        except Exception:
            logger.exception("Erro carregando %s", path)
            return []

    # ---------------------------------------------------------
    # save (batched)
    # ---------------------------------------------------------

    async def save(self, path: Path, data):

        key = self._key(path)

        self._cache[key] = data
        self._dirty.add(key)

        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self):

        try:

            await asyncio.sleep(self._flush_interval)

            dirty = list(self._dirty)
            self._dirty.clear()

            for key in dirty:

                path = Path(key)
                lock = self._get_lock(path)

                async with lock:

                    data = self._cache.get(key, [])

                    await self._atomic_write(path, data)

        finally:

            self._flush_task = None

    async def _atomic_write(self, path: Path, data):

        tmp = path.with_suffix(".tmp")

        try:

            text = json.dumps(data, ensure_ascii=False, indent=2)

            tmp.write_text(text, encoding="utf-8")

            with open(tmp, "r+") as f:
                f.flush()
                import os
                os.fsync(f.fileno())

            tmp.replace(path)

        except Exception:

            logger.exception("Erro salvando %s", path)

            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # ---------------------------------------------------------
    # EVENTOS
    # ---------------------------------------------------------

    async def log_event(
        self,
        text: str,
        *,
        embed_fn: Callable[[str], Awaitable[list]]
    ):

        events = await self.load(self.events_file)

        query = self.alias_resolver.normalize(text)

        vector = await embed_fn(query)

        event = {
            "timestamp": time.time(),
            "text": text,
            "tokens": tokenize(text),
            "vector": vector,
        }

        events.append(event)

        if len(events) > self.MAX_EVENTS:
            events = events[-self.MAX_EVENTS:]

        await self.save(self.events_file, events)

        try:
            update_graph_from_event(text)
        except Exception:
            logger.exception("Falha ao atualizar narrative graph")

        bump_event_version()

    def get_recent_events(self, limit=5):

        key = self._key(self.events_file)

        events = self._cache.get(key, [])

        return [e["text"] for e in events[-limit:]]

    # ---------------------------------------------------------
    # busca semântica
    # ---------------------------------------------------------

    async def search_events(self, query, k, vector_search_fn):

        events = await self.load(self.events_file)

        if not events:
            return []

        return await vector_search_fn(events, query, "text", k)

    async def search_sessions(self, query, k, vector_search_fn):

        sessions = await self.load(self.sessions_file)

        if not sessions:
            return []

        return await vector_search_fn(sessions, query, "summary", k)

    async def search_arcs(self, query, k, vector_search_fn):

        arcs = await self.load(self.arcs_file)

        if not arcs:
            return []

        return await vector_search_fn(arcs, query, "summary", k)

    async def hierarchical_search(self, query, vector_search_fn):

        arcs = await self.search_arcs(query, 2, vector_search_fn)
        sessions = await self.search_sessions(query, 2, vector_search_fn)
        events = await self.search_events(query, 3, vector_search_fn)

        return arcs + sessions + events

    # ---------------------------------------------------------
    # sumarização
    # ---------------------------------------------------------

    def compress_events(self, events):

        size = 0
        selected = []

        for e in reversed(events):

            txt = e["text"]

            size += len(txt)

            if size > self.MAX_EVENT_CHARS:
                break

            selected.append(txt)

        return "\n".join(reversed(selected))

    async def summarize_session(
        self,
        generate_narrative,
        *,
        embed_fn: Callable[[str], Awaitable[list]]
    ):

        events = await self.load(self.events_file)

        if not events:
            return

        text = self.compress_events(events)

        prompt = f"Resuma os principais acontecimentos da sessão:\n{text}"

        summary = await generate_narrative(prompt)

        sessions = await self.load(self.sessions_file)

        session_record = {
            "timestamp": time.time(),
            "summary": summary,
            "tokens": tokenize(summary),
            "vector": await embed_fn(summary),
        }

        sessions.append(session_record)

        if len(sessions) > self.MAX_SESSIONS:
            sessions = sessions[-self.MAX_SESSIONS:]

        await self.save(self.sessions_file, sessions)

        await self.save(self.events_file, [])
