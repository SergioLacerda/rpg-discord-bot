import re


class DynamicContextWindow:

    def __init__(self):

        self.policies = {
            "combat": {"max_docs": 6},
            "dialogue": {"max_docs": 10},
            "investigation": {"max_docs": 18},
            "exploration": {"max_docs": 12},
        }

        self.vocabulary = {

            "combat": {
                "atacar", "ataco", "ataque",
                "golpe", "golpear",
                "tiro", "atirar", "disparo",
                "lutar", "combate",
                "attack", "shoot", "fight",
                "strike", "combat"
            },

            "dialogue": {
                "pergunto", "perguntar",
                "falo", "falar", "digo",
                "respondo", "questiono",
                "ask", "tell", "say",
                "talk", "speak"
            },

            "investigation": {
                "investigo", "investigar",
                "procuro", "buscar",
                "quem", "porque", "por que",
                "evidencia", "pista",
                "investigate", "search",
                "who", "why", "clue",
                "evidence"
            },

            "exploration": {
                "exploro", "explorar",
                "entro", "entro na",
                "vasculho",
                "explore", "enter",
                "look around"
            }
        }

    def tokenize(self, text):

        text = text.lower()

        tokens = re.findall(r"\w+", text)

        return tokens

    def classify(self, query):

        tokens = self.tokenize(query)

        scores = {
            "combat": 0,
            "dialogue": 0,
            "investigation": 0,
            "exploration": 0
        }

        for token in tokens:

            for category, vocab in self.vocabulary.items():

                if token in vocab:
                    scores[category] += 1

        # escolher categoria dominante
        best = max(scores, key=scores.get)

        if scores[best] == 0:
            return "exploration"

        return best

    def select(self, query, docs):

        qtype = self.classify(query)

        policy = self.policies.get(qtype)

        if not policy:
            return docs[:10]

        return docs[:policy["max_docs"]]