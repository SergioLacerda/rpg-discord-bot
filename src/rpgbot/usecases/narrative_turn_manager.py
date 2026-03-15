import asyncio
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class NarrativeTurnManager:
    """
    Controla a ordem narrativa por campanha.

    Resolve problemas de:
    - múltiplas ações simultâneas
    - cancelamento de geração
    - fairness entre jogadores
    - sobrecarga de LLM
    """

    def __init__(self, engine):

        self.engine = engine

        # fila por campanha
        self.queues = defaultdict(asyncio.Queue)

        # tarefa ativa por campanha
        self.active_tasks = {}

        # locks por campanha
        self.locks = defaultdict(asyncio.Lock)

    # ---------------------------------------------------------
    # enqueue action
    # ---------------------------------------------------------

    async def submit_action(
        self,
        campaign_id,
        action,
        ctx,
        index
    ):

        queue = self.queues[campaign_id]

        future = asyncio.get_event_loop().create_future()

        await queue.put({
            "action": action,
            "ctx": ctx,
            "index": index,
            "future": future,
        })

        asyncio.create_task(
            self._process_queue(campaign_id)
        )

        return await future

    # ---------------------------------------------------------
    # queue processor
    # ---------------------------------------------------------

    async def _process_queue(self, campaign_id):

        lock = self.locks[campaign_id]

        if lock.locked():
            return

        async with lock:

            queue = self.queues[campaign_id]

            while not queue.empty():

                job = await queue.get()

                try:

                    result = await self._run_job(
                        campaign_id,
                        job
                    )

                    job["future"].set_result(result)

                except Exception as e:

                    logger.exception("turn_processing_failed")

                    job["future"].set_exception(e)

    # ---------------------------------------------------------
    # run generation
    # ---------------------------------------------------------

    async def _run_job(self, campaign_id, job):

        action = job["action"]
        ctx = job["ctx"]
        index = job["index"]

        msg = await ctx.send("...")

        text = ""
        last_edit = 0

        async for token in self.engine.stream_narrative(
            action,
            index=index
        ):

            text += token

            now = time.time()

            if now - last_edit > 0.8:

                await msg.edit(content=text)

                last_edit = now

        await msg.edit(content=text)

        return text

    # ---------------------------------------------------------
    # cancel campaign
    # ---------------------------------------------------------

    def cancel_campaign(self, campaign_id):

        task = self.active_tasks.get(campaign_id)

        if task and not task.done():

            task.cancel()

            logger.info(
                "Narrative cancelled campaign=%s",
                campaign_id
            )