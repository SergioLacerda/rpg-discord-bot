import asyncio
import logging
from typing import Optional, Callable, Awaitable, AsyncGenerator

from openai import AsyncOpenAI, RateLimitError

from rpgbot.core.container import container
from rpgbot.core.resilience import resilient_call
from rpgbot.infrastructure.narrative_memory import memory
from rpgbot.infrastructure.response_cache import (
    get_cached_response,
    set_cached_response,
)
from rpgbot.usecases.retrieve_context import hierarchical_context
from rpgbot.utils.concurrency.deduplicate_async import InflightDeduplicator
from rpgbot.utils.text.normalize_utils import compress_context
from rpgbot.utils.hash_utils import sha256_hash

logger = logging.getLogger(__name__)

llm_deduplicator = InflightDeduplicator()

MAX_TOKENS = 300
TEMPERATURE = 0.8

class NarrativeEngine:

    def __init__(self):

        self.cancel_flags = {}

    # ---------------------------------------------------------
    # cancelamento
    # ---------------------------------------------------------

    def cancel(self, key: str):

        self.cancel_flags[key] = True

    def _cancelled(self, key):

        return self.cancel_flags.get(key, False)

    # ---------------------------------------------------------
    # memória narrativa
    # ---------------------------------------------------------

    def _get_events(self):

        if hasattr(memory, "get_recent"):
            return memory.get_recent()

        if hasattr(memory, "events"):
            return memory.events

        if hasattr(memory, "history"):
            return memory.history

        return []

    # ---------------------------------------------------------
    # compressão de memória
    # ---------------------------------------------------------

    async def _compress_memory(self, events):

        embed = container.resolve("embedding_provider.embed")

        if len(events) <= 3:
            return events

        kept = []
        vectors = []

        from rpgbot.utils.vector.vector_math import cosine_similarity

        for event in events:

            vec = await embed(event)

            redundant = False

            for v in vectors:

                if cosine_similarity(vec, v) > 0.88:
                    redundant = True
                    break

            if not redundant:

                kept.append(event)
                vectors.append(vec)

        return kept

    # ---------------------------------------------------------
    # prompt builder
    # ---------------------------------------------------------

    async def build_prompt(self, action, ctx_provider, memory_context):

        ctx = await ctx_provider(action)

        ctx = compress_context(ctx)

        history = memory.get()

        return f"""
Você é um mestre de RPG narrativo experiente e imparcial.

Resumo da história até agora:
{history}

Memória recente da sessão:
{memory_context}

Contexto relevante:
{ctx}

Ação do jogador:
{action}

Descreva o que acontece em seguida.
"""

    # ---------------------------------------------------------
    # streaming LLM
    # ---------------------------------------------------------

    async def _stream_llm(
        self,
        prompt: str,
        client: Optional[AsyncOpenAI] = None,
    ) -> AsyncGenerator[str, None]:

        if client:

            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Mestre de RPG narrativo imparcial."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=True,
            )

            async for chunk in stream:

                token = chunk.choices[0].delta.content

                if token:
                    yield token

            return

        # provider genérico

        stream_fn = container.resolve("llm_provider.stream")

        async for token in stream_fn(prompt):

            yield token

    # ---------------------------------------------------------
    # geração streaming
    # ---------------------------------------------------------

    async def stream_narrative(
        self,
        action: str,
        *,
        client: Optional[AsyncOpenAI] = None,
        ctx_provider: Callable[[str], Awaitable[str]] = hierarchical_context,
        index=None,
    ) -> AsyncGenerator[str, None]:

        async def ctx_with_index(query):

            try:
                return await ctx_provider(query, index=index)
            except TypeError:
                return await ctx_provider(query)

        events = self._get_events()

        events = await self._compress_memory(events)

        memory_context = "\n".join(events[-10:])

        prompt = await self.build_prompt(action, ctx_with_index, memory_context)

        key = sha256_hash(prompt)

        # cache

        cached = await get_cached_response(key)

        if cached:

            yield cached
            return

        content = ""

        async for token in self._stream_llm(prompt, client):

            if self._cancelled(key):
                logger.info("narrative cancelled")
                break

            content += token

            yield token

        if content:

            await set_cached_response(key, content)

            if not content.startswith("⚠️"):
                memory.update(content)

    # ---------------------------------------------------------
    # geração padrão (não streaming)
    # ---------------------------------------------------------

    async def generate(
        self,
        action: str,
        *,
        client: Optional[AsyncOpenAI] = None,
        ctx_provider: Callable[[str], Awaitable[str]] = hierarchical_context,
        index=None,
    ) -> str:

        async def run():

            text = ""

            async for token in self.stream_narrative(
                action,
                client=client,
                ctx_provider=ctx_provider,
                index=index,
            ):
                text += token

            return text

        return await llm_deduplicator.run(action, run)