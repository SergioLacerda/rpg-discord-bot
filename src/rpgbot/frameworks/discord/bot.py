import logging
import asyncio
import discord
from discord.ext import commands

from rpgbot.core.config import settings
from rpgbot.usecases.roll_dice import roll_dice
from rpgbot.usecases.generate_npc import generate_npc
from rpgbot.usecases.generate_narrative import generate_narrative
from rpgbot.adapters.storage.file_log_repository import write_log
from rpgbot.adapters.storage.json_session_repository import log_event, summarize_session


# ----------------------------
# logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logging.getLogger("discord").setLevel(logging.WARNING)

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
@commands.cooldown(1, 5, commands.BucketType.user)
async def gm(ctx, *, action: str):

    logger.info("GM command | user=%s | action=%s", ctx.author, action)

    if len(action) > 500:
        await ctx.send("⚠️ A ação é muito longa.")
        return

    try:

        response = await asyncio.wait_for(
            generate_narrative(action),
            timeout=30
        )

        await log_event(action)
        await log_event(response)

        await ctx.send(response)

    except asyncio.TimeoutError:

        await ctx.send("⚠️ O mestre demorou muito para responder.")

    except Exception:

        logger.exception("gm_command_failed")

        await ctx.send("⚠️ O mestre parece confuso por um momento...")


@bot.command()
async def roll(ctx, expr: str):

    logger.info("Roll command | user=%s | expr=%s", ctx.author, expr)

    try:
        result = roll_dice(expr)

        await ctx.send(f"🎲 {result.detail}")

    except ValueError:

        await ctx.send("⚠️ Expressão de dados inválida.")


@bot.command()
async def npc(ctx, *, desc: str):

    logger.info("NPC generation | user=%s", ctx.author)

    npc = await asyncio.to_thread(generate_npc, desc)

    await ctx.send(
        f"**NPC:** {npc['name']}\n"
        f"**Descrição:** {npc['description']}\n"
        f"**Traço:** {npc['trait']}"
    )


@bot.command()
@commands.cooldown(2, 10, commands.BucketType.user)
async def log(ctx, *, text: str):

    logger.info(
        "session_log",
        extra={
            "user": str(ctx.author),
            "channel": str(ctx.channel),
            "text_length": len(text)
        }
    )

    if len(text) > 500:

        await ctx.send("⚠️ O log é muito longo. Limite de 500 caracteres.")
        return

    try:

        file = await write_log(text)

        await ctx.send(f"📜 Evento salvo em `{file.name}`")

    except Exception:

        logger.exception("session_log_failed")

        await ctx.send("⚠️ Não foi possível registrar o evento.")


# ----------------------------
# session management
# ----------------------------

@bot.command()
async def endsession(ctx):

    logger.info("Ending session and summarizing")

    try:

        await asyncio.to_thread(
            summarize_session,
            generate_narrative
        )

        await ctx.send(
            "📚 Sessão resumida e salva na memória da campanha."
        )

    except Exception:

        logger.exception("session_summary_failed")

        await ctx.send("⚠️ Falha ao resumir a sessão.")


# ----------------------------
# message handler
# ----------------------------

@bot.event
async def on_message(message):

    # ignora mensagens do próprio bot
    if message.author == bot.user:
        return

    # # ignora DMs -> mensagens em privado
    # if message.guild is None:
    #     return

    # processa comandos
    await bot.process_commands(message)


# ----------------------------
# error handler
# ----------------------------

@bot.event
async def on_command_error(ctx, error):

    if isinstance(error, commands.CommandOnCooldown):

        await ctx.send("⏳ Aguarde alguns segundos antes de usar novamente.")
        return

    logger.exception("discord_command_error", exc_info=error)


@bot.command()
async def ping(ctx):
    await ctx.send("pong")

# ----------------------------
# start bot
# ----------------------------

def main():

    logger.info("Starting RPG bot")

    bot.run(settings.DISCORD_TOKEN)


if __name__ == "__main__":
    main()