
class LexicalGraphMerge:

    def __init__(self, lexical_fn, graph_fn, fusion_fn):
        self.lexical_fn = lexical_fn
        self.graph_fn = graph_fn
        self.fusion_fn = fusion_fn

    def run(self, ctx, candidates):

        lexical = self.lexical_fn(ctx.query_tokens)
        graph_docs = self.graph_fn(ctx.query_tokens)

        if lexical or graph_docs:

            return self.fusion_fn(
                candidates[:100] + graph_docs[:100],
                lexical[:100] if lexical else [],
                limit=150
            )

        return candidates