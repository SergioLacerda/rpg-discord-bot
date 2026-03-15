import pkgutil
import importlib
import inspect
import re


def _snake(name):

    return re.sub(r'(?<!^)(?=[A-Z])', "_", name).lower()


def register_fake_providers(container):

    for _, module_name, _ in pkgutil.iter_modules(["tests/fixtures"]):

        if module_name == "autoload":
            continue

        module = importlib.import_module(f"tests.fixtures.{module_name}")

        for name, obj in inspect.getmembers(module, inspect.isclass):

            if not name.startswith("Fake"):
                continue

            service = _snake(name.replace("Fake", ""))

            container.register(
                service,
                lambda cls=obj: cls(),
                singleton=True
            )
