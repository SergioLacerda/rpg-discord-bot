class ScopedContainer:

    def __init__(self, parent):
        self.parent = parent
        self._instances = {}

    def resolve(self, name):

        if name in self._instances:
            return self._instances[name]

        provider, singleton = self.parent._providers[name]

        instance = self.parent._build(provider)

        # scoped singleton
        if singleton:
            self._instances[name] = instance

        return instance