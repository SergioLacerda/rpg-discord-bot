# 🎲 AI RPG Discord Bot

Bot de RPG narrativo para Discord com **IA como mestre de jogo**,
memória persistente de campanha e sistema de rolagem de dados avançado.

## ✨ Funcionalidades

- 🤖 Narrativa automática via IA
- 🎲 Sistema de rolagem de dados
- 📚 Memória vetorial da campanha (RAG)
- 🧠 Memória episódica de sessões
- 💾 NPCs persistentes
- 🧪 Testes automatizados

------------------------------------------------------------------------

## 📦 Estrutura do projeto

campaign/
   estado/
   faccoes/
   lore/
   memory/
   npc/
   sessions/

src/rpgbot/
   adapters/
       embeddings/
       llm/
       rag/
       storage/
   ai/
   core/
   domain/
       dice/
       entities/
       value_objects/
   frameworks/
       discord/
   bot.py

Arquivos de persistência:
campaign/memory/events.json
campaign/memory/response_cache.json
campaign/memory/embedding_cache.json
campaign/index_vectors.json

Configuração:
pyproject.toml
.env.test
------------------------------------------------------------------------

## 🔑 Configuração de chaves

## Discord Bot Token

1. Acesse: <https://discord.com/developers/applications>
2. Clique em **New Application**
3. Vá em **Bot → Add Bot**
4. Copie o **Bot Token**

Exemplo:

DISCORD_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxx

------------------------------------------------------------------------

## OpenAI API Key

1. Acesse: <https://platform.openai.com/api-keys>

2. Clique em **Create new secret key**

3. Copie a chave gerada.

Exemplo:

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

------------------------------------------------------------------------

## ⚙️ Configuração do ambiente

## Criar ambiente virtual

Linux / Mac:

```python -m venv .venv```

Windows:

```python -m venv .venv```

------------------------------------------------------------------------

## Ativar ambiente virtual

Linux / Mac:

```source .venv/bin/activate```

Windows:

```.venv`\Scripts`{=tex}`\activate`{=tex}```

------------------------------------------------------------------------

## 📥 Instalar dependências

pip install -e .

pip install -e ".[local]"

------------------------------------------------------------------------

## 🔐 Variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```DISCORD_TOKEN=your_discord_token OPENAI_API_KEY=your_openai_key```

------------------------------------------------------------------------

## 🚀 Executar o bot

rpgbot

------------------------------------------------------------------------

## 🎲 Comandos do bot

```!roll 1d20\```
```!roll 4d6dl1\```
```!gm investigo o armazém abandonado\```
```!npc mercenário cansado de guerra```

------------------------------------------------------------------------

## 🧪 Testes

Executar testes:

pytest

Cobertura:

pytest --cov=rpgbot --cov-report=html

------------------------------------------------------------------------

## 🧠 Sistema de memória

Arc memory → história da campanha\
Session memory → sessões anteriores\
Event memory → eventos recentes

Fluxo:

query → arc → session → events

------------------------------------------------------------------------
