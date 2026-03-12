import asyncio
import logging
import hashlib
from typing import Optional

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

from rpgbot.config import OPENAI_API_KEY
from rpgbot.services.memory_service import hierarchical_context
from rpgbot.infrastructure.embedding_client import get_client

MODEL = "gpt-4o-mini"
MAX_TOKENS = 300
TEMPERATURE = 0.8
TIMEOUT = 20
MAX_RETRIES = 3
RESPONSE_CACHE: dict[str, str] = {}

logger = logging.getLogger(__name__)


async def build_prompt(player_action: str) -> str:
    context = await hierarchical_context(player_action)
    logger.debug(f"Contexto recuperado ({len(context)} caracteres) para ação: {player_action[:50]}...")

    return f"""
Você é um mestre de RPG narrativo experiente e imparcial.
Regras obrigatórias:
- Descreva apenas o ambiente, consequências e reações do mundo/NPCs
- Avance a narrativa de forma fluida e imersiva
- NUNCA controle, decida ou fale pelos personagens dos jogadores
- Seja conciso: 3 a 6 frases no máximo
- Mantenha tom épico, coerente e adequado ao gênero RPG

Contexto relevante da campanha:
{context}

Ação declarada pelo jogador:
{player_action}

Descreva o que acontece em seguida.
"""


async def generate_narrative(player_action: str) -> str:
    prompt = await build_prompt(player_action)
    cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    if cache_key in RESPONSE_CACHE:
        logger.debug(f"Resposta encontrada no cache para ação: {player_action[:30]}...")
        return RESPONSE_CACHE[cache_key]

    client: AsyncOpenAI = await get_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Gerando narrativa | tentativa {attempt}/{MAX_RETRIES} | ação={player_action[:60]}...")

            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Você é um mestre de RPG narrativo imparcial."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=TIMEOUT,
            )

            content = response.choices[0].message.content.strip()

            if not content:
                logger.warning("Resposta vazia recebida da OpenAI")
                raise ValueError("Resposta vazia")

            RESPONSE_CACHE[cache_key] = content

            return content

        except RateLimitError as e:
            wait = 2 ** attempt  # 2s → 4s → 8s...
            logger.warning(f"Rate limit atingido | tentativa {attempt} → aguardando {wait}s")
            await asyncio.sleep(wait)

        except APITimeoutError:
            logger.warning(f"Timeout na API | tentativa {attempt}")
            await asyncio.sleep(1.5)

        except APIError as e:
            logger.error(f"Erro da API OpenAI | tentativa {attempt}: {str(e)}")
            await asyncio.sleep(1)

        except Exception as e:
            logger.exception(f"Erro inesperado na geração de narrativa | tentativa {attempt}")
            await asyncio.sleep(1)

    logger.error("Falha definitiva após todas as tentativas de gerar narrativa")

    return (
        "O ambiente parece congelado por um momento... como se o destino estivesse "
        "hesitando. O que você decide fazer agora?"
    )