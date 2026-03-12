import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pytest


@pytest.fixture(scope="session", autouse=True)
def load_test_env():
    """
    Carrega automaticamente o arquivo .env.test na raiz do projeto
    antes de qualquer teste ser executado.
    """
    # Procura .env.test na raiz do projeto
    env_file = find_dotenv(".env.test", raise_error_if_not_found=False)

    if env_file:
        load_dotenv(env_file, override=True)
        print(f"[TEST ENV] Carregado com sucesso: {env_file}")
    else:
        print("[TEST ENV] Arquivo .env.test não encontrado → usando variáveis de ambiente existentes")

    # Opcional: validação mínima (útil para evitar surpresas)
    if not os.getenv("OPENAI_API_KEY"):
        print("[TEST ENV] AVISO: OPENAI_API_KEY não definido no ambiente de teste")