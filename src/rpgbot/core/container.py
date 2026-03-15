import asyncio
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

        self._compiled_builders = {}
        self._compiled_graphs = {}

        self._bootstrapped = False

    # --------------------------------------------------
    # BOOTSTRAP
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

            if name in providers:
                deps.append((name, name))
                continue

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
    # COMPILED BUILDER
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
    # COMPILE GRAPH
    # --------------------------------------------------

    def _compile_graph(self, root):

        if root in self._compiled_graphs:
            return self._compiled_graphs[root]

        order = []
        visited = set()
        stack = set()

        def visit(service):

            if service in stack:
                raise RuntimeError(f"Circular dependency detected: {service}")

            if service in visited:
                return

            stack.add(service)
            visited.add(service)

            config = self._providers.get(service)

            if not config:
                raise KeyError(f"Service '{service}' not registered")

            provider = config["provider"]

            if callable(provider):

                deps = self._compile_build_plan(provider)

                for _, dep in deps:
                    visit(dep)

            stack.remove(service)
            order.append(service)

        visit(root)

        self._compiled_graphs[root] = order

        return order

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
        self._compiled_graphs.clear()
        self._build_plan_cache.clear()

    # --------------------------------------------------
    # ATTRIBUTE PATH
    # --------------------------------------------------

    def _resolve_attr(self, name):

        parts = name.split(".")

        base = parts[0]

        # fallback automático: service.method -> method
        if base not in self._providers and parts[-1] in self._providers:
            return self.resolve(parts[-1])

        obj = self.resolve(base)

        for attr in parts[1:]:
            obj = getattr(obj, attr)

        return obj

    # --------------------------------------------------
    # RESOLVE
    # --------------------------------------------------

    def resolve(self, name):

        if "." in name:
            return self._resolve_attr(name)

        self._ensure_bootstrap()

        if name not in self._providers:
            raise KeyError(f"Service '{name}' not registered")

        # weak cache
        if name in self._weak_instances:
            return self._weak_instances[name]

        # singleton cache
        if name in self._instances:

            instance, expires = self._instances[name]

            if expires and time.time() > expires:
                del self._instances[name]
            else:
                return instance

        config = self._providers[name]

        singleton = config["singleton"]
        weak = config["weak"]
        ttl = config["ttl"]

        instance = self.resolve_graph(name)

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
    # GRAPH RESOLUTION
    # --------------------------------------------------

    def resolve_graph(self, root):

        order = self._compile_graph(root)

        instances = {}

        for svc in order:

            config = self._providers[svc]

            if config["singleton"] and svc in self._instances:
                instances[svc] = self._instances[svc][0]
                continue

            builder = self._compile_builder(svc)

            instance = builder()

            instances[svc] = instance

        return instances[root]

    # --------------------------------------------------
    # ASYNC RESOLVE
    # --------------------------------------------------

    async def resolve_async(self, name):

        if "." in name:
            return self._resolve_attr(name)

        order = self._compile_graph(name)

        tasks = {}

        async def build_service(svc):

            config = self._providers[svc]

            if config["singleton"] and svc in self._instances:
                return self._instances[svc][0]

            builder = self._compile_builder(svc)

            instance = builder()

            if inspect.isawaitable(instance):
                instance = await instance

            if config["singleton"]:
                self._instances[svc] = (instance, None)

            return instance

        for svc in order:

            provider = self._providers[svc]["provider"]

            deps = []

            if callable(provider):
                deps = [d for _, d in self._compile_build_plan(provider)]

            async def run(svc=svc, deps=deps):

                for d in deps:
                    await tasks[d]

                return await build_service(svc)

            tasks[svc] = asyncio.create_task(run())

        return await tasks[name]

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
        self._compiled_graphs.clear()
        self._build_plan_cache.clear()

    # --------------------------------------------------
    # SCOPE
    # --------------------------------------------------

    def scope(self):
        return ScopedContainer(self)


# --------------------------------------------------
# SCOPED CONTAINER
# --------------------------------------------------

class ScopedContainer:

    def __init__(self, parent):

        self.parent = parent

        self._providers = parent._providers
        self._instances = {}
        self._weak_instances = weakref.WeakValueDictionary()

    def resolve(self, name):

        if "." in name:
            parts = name.split(".")
            obj = self.resolve(parts[0])
            for attr in parts[1:]:
                obj = getattr(obj, attr)
            return obj

        if name in self._weak_instances:
            return self._weak_instances[name]

        if name in self._instances:
            return self._instances[name]

        if name not in self._providers:
            raise KeyError(f"Service '{name}' not registered")

        config = self._providers[name]

        provider = config["provider"]
        singleton = config["singleton"]
        weak = config["weak"]

        if callable(provider):

            sig = inspect.signature(provider)

            kwargs = {}

            for param in sig.parameters:
                kwargs[param] = self.resolve(param)

            instance = provider(**kwargs)

        else:
            instance = provider

        if singleton:

            if weak:
                self._weak_instances[name] = instance
            else:
                self._instances[name] = instance

        return instance

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):

        self._instances.clear()
        self._weak_instances.clear()


container = Container()
