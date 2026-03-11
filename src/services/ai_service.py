import time
import logging

from openai import APITimeoutError, APIError, RateLimitError

from src.services.memory_service import build_context
from src.infrastructure.embedding_client import get_client


MODEL = "gpt-4o-mini"
MAX_TOKENS = 300
TEMPERATURE = 0.8
TIMEOUT = 20
MAX_RETRIES = 3


logger = logging.getLogger(__name__)


def build_prompt(player_action: str):

    context = build_context(player_action)

    logger.debug(f"Context length: {len(context)} chars")

    return f"""
Você é um mestre de RPG narrativo.

Regras:

- descreva o ambiente
- avance a narrativa
- não controle diretamente os jogadores
- seja conciso (3-6 frases)

Contexto relevante da campanha:

{context}

Ação do jogador:

{player_action}

Descreva o que acontece a seguir.
"""


def generate_narrative(player_action: str):

    prompt = build_prompt(player_action)

    client = get_client()

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            logger.info(
                f"Gerando narrativa | tentativa={attempt} | ação={player_action}"
            )

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Você é um mestre de RPG."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=TIMEOUT
            )

            content = response.choices[0].message.content

            if not content or not content.strip():

                logger.warning("Resposta da IA veio vazia")

                raise ValueError("Resposta vazia da IA")

            return content.strip()

        except RateLimitError:

            wait = 2 ** (attempt - 1)

            logger.warning(
                f"Rate limit atingido | tentativa={attempt} | retry em {wait}s"
            )

            time.sleep(wait)

        except APITimeoutError:

            logger.warning(f"Timeout da IA | tentativa={attempt}")

            time.sleep(1)

        except APIError as e:

            logger.error(
                f"Erro da API OpenAI | tentativa={attempt} | erro={e}"
            )

            time.sleep(1)

        except Exception as e:

            logger.exception(
                f"Erro inesperado ao gerar narrativa | tentativa={attempt} | erro={e}"
            )

            time.sleep(1)

    logger.error("Falha definitiva ao gerar narrativa após todas as tentativas")

    return (
        "⚠️ O ambiente permanece silencioso por um instante, "
        "como se algo estivesse prestes a acontecer."
    )