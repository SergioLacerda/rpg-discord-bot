import inspect


class RetrievalPipeline:

    def __init__(
        self,
        recall,
        expansion,
        filtering,
        hierarchical=None,
        max_candidates=200,
        early_stop=40
    ):

        self.recall = recall or []
        self.expansion = expansion or []
        self.filtering = filtering or []

        # novo
        self.hierarchical = hierarchical or []

        self.max_candidates = max_candidates
        self.early_stop = early_stop


    async def _run_stage(self, stage, ctx, candidates):

        result = stage.run(ctx, candidates)

        if inspect.isawaitable(result):
            result = await result

        return result if result is not None else candidates


    def _should_run(self, stage, candidates):

        min_required = getattr(stage, "min_candidates", 0)

        if len(candidates) < min_required:
            return False

        return True


    async def _run_group(self, stages, ctx, candidates):

        stages = sorted(
            stages,
            key=lambda s: getattr(s, "priority", 100)
        )

        for stage in stages:

            if not self._should_run(stage, candidates):
                continue

            candidates = await self._run_stage(stage, ctx, candidates)

            if len(candidates) > self.max_candidates:
                candidates = candidates[: self.max_candidates]

            if len(candidates) >= self.early_stop:
                break

        return candidates


    async def run(self, ctx):

        candidates = []

        # -----------------------------
        # hierarchical recall (novo)
        # -----------------------------

        if self.hierarchical:

            candidates = await self._run_group(
                self.hierarchical,
                ctx,
                candidates
            )

            if len(candidates) >= self.early_stop:
                return candidates

        # -----------------------------
        # recall
        # -----------------------------

        candidates = await self._run_group(
            self.recall,
            ctx,
            candidates
        )

        # -----------------------------
        # expansion
        # -----------------------------

        candidates = await self._run_group(
            self.expansion,
            ctx,
            candidates
        )

        # -----------------------------
        # filtering
        # -----------------------------

        candidates = await self._run_group(
            self.filtering,
            ctx,
            candidates
        )

        return candidates