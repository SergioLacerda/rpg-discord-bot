
class RoutingStage:

    def __init__(self, router):
        self.router = router

    def run(self, ctx, candidates):

        return self.router.route_retrieval(
            ctx.query_type,
            candidates
        )