import inspect


class Container:

    def __init__(self):
        self._providers = {}
        self._instances = {}

    def register(self, name, provider, singleton=True):
        self._providers[name] = (provider, singleton)

    def resolve(self, name):

        if name in self._instances:
            return self._instances[name]

        if name not in self._providers:
            raise KeyError(f"Service '{name}' not registered")

        provider, singleton = self._providers[name]

        instance = self._build(provider)

        if singleton:
            self._instances[name] = instance

        return instance

    # -----------------------------
    # AUTO-WIRING
    # -----------------------------

    def _build(self, provider):

        if not callable(provider):
            return provider

        sig = inspect.signature(provider)

        kwargs = {}

        for name in sig.parameters:

            if name in self._providers:
                kwargs[name] = self.resolve(name)

        return provider(**kwargs)

    def reset(self):
        self._providers.clear()
        self._instances.clear()


container = Container()