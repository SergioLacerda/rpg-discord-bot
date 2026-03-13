from rpgbot.core.container import container


class LazyService:
    """
    Proxy que resolve o serviço no container apenas
    quando algum atributo é acessado.
    """

    def __init__(self, name):
        self.name = name

    def _resolve(self):
        try:
            return container.resolve(self.name)
        except KeyError:
            raise RuntimeError(
                f"Service '{self.name}' not registered. Did you call setup_container()?"
            )

    def __getattr__(self, attr):
        service = self._resolve()
        return getattr(service, attr)


def lazy(name):
    return LazyService(name)