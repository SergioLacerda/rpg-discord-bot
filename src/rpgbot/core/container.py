import inspect
import time
import weakref


class Container:

    def __init__(self):

        self._providers = {}
        self._instances = {}
        self._weak_instances = weakref.WeakValueDictionary()

        self._resolving = set()

        self._signature_cache = {}
        self._build_plan_cache = {}

        # nova estrutura
        self._compiled_builders = {}

        self._bootstrapped = False

    # --------------------------------------------------
    # BUILD
    # --------------------------------------------------

    def _build(self, provider):

        if not callable(provider):
            return provider

        plan = self._build_plan_cache.get(provider)

        if plan is None:

            sig = inspect.signature(provider)
            providers = self._providers

            plan = []

            for name, param in sig.parameters.items():

                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD
                ):
                    continue

                if name in providers:
                    plan.append((name, name))
                    continue

                ann = param.annotation

                if ann is inspect._empty:
                    continue

                for svc_name, svc in providers.items():

                    svc_provider = svc["provider"]

                    if svc_provider is ann:
                        plan.append((name, svc_name))
                        break

                    if inspect.isclass(svc_provider) and issubclass(svc_provider, ann):
                        plan.append((name, svc_name))
                        break

            self._build_plan_cache[provider] = plan

        kwargs = {}

        for param_name, service_name in plan:
            kwargs[param_name] = self.resolve(service_name)

        return provider(**kwargs)

    # --------------------------------------------------
    # LAZY BOOTSTRAP
    # --------------------------------------------------

    def _ensure_bootstrap(self):

        if self._bootstrapped:
            return

        try:
            from rpgbot.bootstrap import setup_container
            setup_container()
        except Exception:
            pass

        self._bootstrapped = True

    # --------------------------------------------------
    # SIGNATURE CACHE
    # --------------------------------------------------

    def _get_signature(self, provider):

        sig = self._signature_cache.get(provider)

        if sig is None:
            sig = inspect.signature(provider)
            self._signature_cache[provider] = sig

        return sig

    # --------------------------------------------------
    # BUILD PLAN
    # --------------------------------------------------

    def _compile_build_plan(self, provider):

        plan = self._build_plan_cache.get(provider)

        if plan:
            return plan

        sig = self._get_signature(provider)

        providers = self._providers

        deps = []

        for name, param in sig.parameters.items():

            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD
            ):
                continue

            # match por nome
            if name in providers:
                deps.append((name, name))
                continue

            # match por tipo
            ann = param.annotation

            if ann is inspect._empty:
                continue

            for svc_name, cfg in providers.items():

                svc_provider = cfg["provider"]

                if svc_provider is ann:
                    deps.append((name, svc_name))
                    break

                if inspect.isclass(svc_provider) and issubclass(svc_provider, ann):
                    deps.append((name, svc_name))
                    break

        self._build_plan_cache[provider] = deps

        return deps

    # --------------------------------------------------
    # GRAPH COMPILED BUILDER
    # --------------------------------------------------

    def _compile_builder(self, name):

        if name in self._compiled_builders:
            return self._compiled_builders[name]

        config = self._providers[name]
        provider = config["provider"]

        if not callable(provider):

            def builder():
                return provider

            self._compiled_builders[name] = builder
            return builder

        plan = self._compile_build_plan(provider)

        def builder():

            kwargs = {}

            for param_name, dep in plan:
                kwargs[param_name] = self.resolve(dep)

            return provider(**kwargs)

        self._compiled_builders[name] = builder

        return builder

    # --------------------------------------------------
    # REGISTER
    # --------------------------------------------------

    def register(
        self,
        name,
        provider,
        *,
        singleton=True,
        weak=False,
        ttl=None,
    ):

        self._providers[name] = {
            "provider": provider,
            "singleton": singleton,
            "weak": weak,
            "ttl": ttl,
        }

        # invalidar caches
        self._compiled_builders.pop(name, None)

    # --------------------------------------------------
    # RESOLVE
    # --------------------------------------------------

    def resolve(self, name):

        if name not in self._providers:

            from rpgbot.bootstrap import setup_container

            setup_container()

        if name not in self._providers:
            raise KeyError(f"Service '{name}' not registered")

            
        if name not in self._providers:
            raise KeyError(f"Service '{name}' not registered")

        if name in self._weak_instances:
            return self._weak_instances[name]

        if name in self._instances:
            instance, expires = self._instances[name]

            if expires and time.time() > expires:
                del self._instances[name]
            else:
                return instance

        if name in self._resolving:
            raise RuntimeError(f"Circular dependency detected: {name}")

        config = self._providers[name]

        provider = config["provider"]
        singleton = config["singleton"]
        weak = config["weak"]
        ttl = config["ttl"]

        self._resolving.add(name)

        try:

            instance = self._build(provider)

        finally:
            # 🔑 garante limpeza mesmo com exceção
            self._resolving.remove(name)

        if singleton:

            if weak:
                self._weak_instances[name] = instance

            else:

                expires = None
                if ttl:
                    expires = time.time() + ttl

                self._instances[name] = (instance, expires)

        return instance

    # --------------------------------------------------
    # ASYNC RESOLVE
    # --------------------------------------------------

    async def resolve_async(self, name):

        instance = self.resolve(name)

        if inspect.isawaitable(instance):
            instance = await instance

        return instance

    # --------------------------------------------------
    # FUNCTION INJECTION
    # --------------------------------------------------

    def inject(self, fn):

        sig = inspect.signature(fn)

        def wrapper(*args, **kwargs):

            for name in sig.parameters:

                if name not in kwargs and name in self._providers:
                    kwargs[name] = self.resolve(name)

            return fn(*args, **kwargs)

        return wrapper

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset_instances(self):
        self._instances.clear()
        self._weak_instances.clear()

    def reset(self):
        self._providers.clear()
        self.reset_instances()
        self._compiled_builders.clear()

    # --------------------------------------------------
    # SCOPE
    # --------------------------------------------------

    def scope(self):
        return ScopedContainer(self)


container = Container()