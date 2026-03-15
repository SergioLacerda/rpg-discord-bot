from typing import Callable, Dict, Any
import difflib
import importlib.metadata


class ProviderRegistry:

    def __init__(self):

        self._providers: Dict[str, Callable[..., Any]] = {}
        self._aliases: Dict[str, str] = {}

    # ---------------------------------------------------------
    # register provider
    # ---------------------------------------------------------

    def register(self, name: str, aliases: list[str] | None = None):

        name = name.lower().strip()

        def decorator(factory: Callable[..., Any]):

            self._providers[name] = factory

            if aliases:
                for alias in aliases:
                    self._aliases[alias.lower()] = name

            return factory

        return decorator

    # ---------------------------------------------------------
    # plugin loader
    # ---------------------------------------------------------

    def load_plugins(self):

        try:
            eps = importlib.metadata.entry_points(group="rpgbot.providers")
        except TypeError:
            eps = importlib.metadata.entry_points().get("rpgbot.providers", [])

        for ep in eps:

            try:
                register_fn = ep.load()

                # plugin recebe registry e registra providers
                register_fn(self)

            except Exception as e:
                print(f"[ProviderRegistry] Failed to load plugin {ep.name}: {e}")

    # ---------------------------------------------------------
    # create provider
    # ---------------------------------------------------------

    def create(self, name: str, **kwargs):

        name = name.lower().strip()
        name = self._aliases.get(name, name)

        factory = self._providers.get(name)

        if not factory:

            available = sorted(self._providers.keys())

            suggestion = difflib.get_close_matches(name, available, n=1)

            msg = f"Provider '{name}' não registrado."

            if suggestion:
                msg += f" Did you mean '{suggestion[0]}'?"

            msg += f" Disponíveis: {', '.join(available)}"

            raise RuntimeError(msg)

        return factory(**kwargs)

    # ---------------------------------------------------------

    def list(self):

        return sorted(self._providers.keys())