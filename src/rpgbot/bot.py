import logging
import asyncio
import discord
from discord.ext import commands

from rpgbot.config import DISCORD_TOKEN

from rpgbot.services.dice_service import roll_dice
from rpgbot.services.log_service import write_log
from rpgbot.services.npc_service import generate_npc
from rpgbot.services.session_memory import log_event, summarize_session
from rpgbot.services.ai_service import generate_narrative
from rpgbot.utils.async_tools import run_blocking

# ----------------------------
# logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logging.getLogger("discord").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


# ----------------------------
# bot setup
# ----------------------------

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


# ----------------------------
# commands
# ----------------------------

@bot.command()
async def gm(ctx, *, action):

    logger.info(f"GM command | user={ctx.author} | action={action}")

    try:

        # roda IA fora do loop async
        response = await run_blocking(generate_narrative, action)

        await log_event(action)
        await log_event(response)

        await ctx.send(response)

    except Exception:

        logger.exception("Erro no comando GM")

        await ctx.send("⚠️ O mestre parece confuso por um momento...")


@bot.command()
async def roll(ctx, expr: str):

    logger.info(f"Roll command | user={ctx.author} | expr={expr}")

    try:

        result = roll_dice(expr)

        logger.info(f"Dice result | {result.detail}")

        await ctx.send(f"🎲 {result.detail}")

    except ValueError:

        await ctx.send("⚠️ Expressão de dados inválida.")


@bot.command()
async def npc(ctx, *, desc):

    logger.info(f"NPC generation | user={ctx.author}")

    npc = generate_npc(desc)

    await ctx.send(
        f"**NPC:** {npc['name']}\n"
        f"**Descrição:** {npc['description']}\n"
        f"**Traço:** {npc['trait']}"
    )


@bot.command()
async def log(ctx, *, text):

    logger.info(f"Session log | user={ctx.author} | text={text}")

    file = write_log(text)

    await ctx.send(f"📜 Evento salvo em `{file}`")


# ----------------------------
# session management
# ----------------------------

@bot.command()
async def endsession(ctx):

    logger.info("Ending session and summarizing")

    try:

        await asyncio.to_thread(summarize_session, generate_narrative)

        await ctx.send("📚 Sessão resumida e salva na memória da campanha.")

    except Exception:

        logger.exception("Erro ao resumir sessão")

        await ctx.send("⚠️ Falha ao resumir a sessão.")

@bot.event
async def on_message(message):

    # evita loop infinito
    if message.author.bot:
        return

    await bot.process_commands(message)

@bot.command(name="rebuild-index")
@commands.has_permissions(administrator=True)
async def rebuild_index_command(ctx):
    await ctx.send("🔄 Iniciando rebuild-index... Isso pode demorar alguns segundos.")
    
    try:
        await rebuild_index()

        await ctx.send("✅ Reindexação concluída com sucesso!")
    except Exception as e:
        logger.exception("Erro durante rebuild-index")
        await ctx.send(f"❌ Erro durante reindexação: {str(e)}")


# ----------------------------
# start bot
# ----------------------------

logger.info("Starting RPG bot")

bot.run(DISCORD_TOKEN)

def main():
    bot.run(DISCORD_TOKEN)