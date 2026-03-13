import logging
import asyncio
import discord
from discord.ext import commands

from rpgbot.adapters.storage.file_log_repository import write_log
from rpgbot.adapters.storage.json_session_repository import log_event, summarize_session

from rpgbot.core.config import settings
from rpgbot.core.container import container

from rpgbot.usecases.retrieve_context import get_campaign_index
from rpgbot.usecases.roll_dice import roll_dice
from rpgbot.usecases.generate_npc import generate_npc
from rpgbot.usecases.generate_narrative import generate_narrative


# ----------------------------
# logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logging.getLogger("discord").setLevel(logging.WARNING)

logger = logging.getLogger("rpgbot.discord")


# ----------------------------
# bot setup
# ----------------------------

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


# ---------------------------------
# scene context reuse
# ---------------------------------

SCENE_CONTEXT = {}
SCENE_TTL = 20

# ---------------------------------
# scene context reuse
# ---------------------------------

SCENE_CONTEXT = {}
SCENE_TTL = 20


async def warmup():

    await vector_index.ensure_ann_ready()



def get_scene_context(campaign_id):

    entry = SCENE_CONTEXT.get(campaign_id)

    if not entry:
        return None

    if (time.time() - entry["ts"]) > SCENE_TTL:
        return None

    return entry["context"]


def update_scene_context(campaign_id, context):

    SCENE_CONTEXT[campaign_id] = {
        "context": context,
        "ts": time.time()
    }

# ----------------------------
# services
# ----------------------------

vector_index = build_vector_index()
retrieval_engine = container.resolve("retrieval_engine")


# ----------------------------
# commands
# ----------------------------

@bot.command()
@commands.cooldown(1, 5, commands.BucketType.user)
async def gm(ctx, *, action: str):

    logger.info("gm_command user=%s action=%s", ctx.author, action)

    if len(action) > 500:
        await ctx.send("⚠️ A ação é muito longa.")
        return

    campaign_context = container.resolve("campaign_context")
    campaign_id = ctx.guild.id if ctx.guild else "dm"

    try:

        async with ctx.typing():

            with campaign_context.scope(campaign_id):

                index = get_campaign_index(campaign_id)

                context = get_scene_context(campaign_id)

                if not context:

                    context = await retrieval_engine.search(
                        action,
                        campaign_id=campaign_id
                    )

                    update_scene_context(campaign_id, context)

                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            generate_narrative,
                            action,
                            index=index,
                            context=context
                        ),
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


# ----------------------------
# dice
# ----------------------------

@bot.command()
async def roll(ctx, expr: str):

    logger.info("roll_command user=%s expr=%s", ctx.author, expr)

    try:

        result = await asyncio.to_thread(roll_dice, expr)

        await ctx.send(f"🎲 {result.detail}")

    except ValueError:

        await ctx.send("⚠️ Expressão de dados inválida.")


# ----------------------------
# npc
# ----------------------------

@bot.command()
async def npc(ctx, *, desc: str):

    logger.info("npc_generation user=%s", ctx.author)

    async with ctx.typing():

        npc = await asyncio.to_thread(generate_npc, desc)

    await ctx.send(
        f"**NPC:** {npc['name']}\n"
        f"**Descrição:** {npc['description']}\n"
        f"**Traço:** {npc['trait']}"
    )


# ----------------------------
# ask (RAG)
# ----------------------------

@bot.command()
async def ask(ctx, *, question):

    logger.info("ask user=%s question=%s", ctx.author, question)

    async with ctx.typing():

        result = await retrieval_engine.search(
            question,
            campaign_id=ctx.guild.id if ctx.guild else "dm"
        )

    if result:
        await ctx.send(result[0])
    else:
        await ctx.send("Não encontrei nada relevante.")


# ----------------------------
# session log
# ----------------------------

@bot.command()
@commands.cooldown(2, 10, commands.BucketType.user)
async def log(ctx, *, text: str):

    logger.info(
        "session_log user=%s channel=%s length=%d",
        ctx.author,
        ctx.channel,
        len(text)
    )

    if len(text) > 500:

        await ctx.send("⚠️ O log é muito longo. Limite de 500 caracteres.")
        return

    try:

        file = await asyncio.to_thread(write_log, text)

        await ctx.send(f"📜 Evento salvo em `{file.name}`")

    except Exception:

        logger.exception("session_log_failed")

        await ctx.send("⚠️ Não foi possível registrar o evento.")


# ----------------------------
# end session
# ----------------------------

@bot.command()
async def endsession(ctx):

    campaign_context = container.resolve("campaign_context")
    campaign_id = ctx.guild.id if ctx.guild else "dm"

    try:

        async with ctx.typing():

            with campaign_context.scope(campaign_id):

                await asyncio.to_thread(
                    summarize_session,
                    generate_narrative
                )

        await ctx.send("📚 Sessão resumida.")

    except Exception:

        logger.exception("session_summary_failed")

        await ctx.send("⚠️ Não consegui resumir a sessão.")


# ----------------------------
# message handler
# ----------------------------

@bot.event
async def on_message(message):

    if message.author.bot:
        return

    await bot.process_commands(message)


# ----------------------------
# errors
# ----------------------------

@bot.event
async def on_command_error(ctx, error):

    if isinstance(error, commands.CommandOnCooldown):

        await ctx.send("⏳ Aguarde alguns segundos antes de usar novamente.")
        return

    logger.exception(
        "discord_command_error user=%s command=%s",
        ctx.author,
        ctx.command
    )


# ----------------------------
# ping
# ----------------------------

@bot.command()
async def ping(ctx):

    await ctx.send("pong")


# ----------------------------
# start bot
# ----------------------------

def main():

    bot.loop.create_task(warmup())

    logger.info("Starting RPG bot")

    bot.run(settings.DISCORD_TOKEN)


if __name__ == "__main__":
    main()