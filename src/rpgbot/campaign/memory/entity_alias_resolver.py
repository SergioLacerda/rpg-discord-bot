class EntityAliasResolver:

    def __init__(self, alias_map=None):

        self.alias_map = alias_map or {}

    def normalize(self, text):

        text = text.lower()

        for canonical, aliases in self.alias_map.items():

            for alias in aliases:

                if alias in text:
                    text = text.replace(alias, canonical)

        return text