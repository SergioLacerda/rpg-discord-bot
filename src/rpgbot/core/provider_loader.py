import importlib
import pkgutil


def load_providers(package_name: str):

    package = importlib.import_module(package_name)

    for _, module_name, ispkg in pkgutil.iter_modules(package.__path__):

        # ignorar subpacotes
        if ispkg:
            continue

        # ignorar arquivos que não são providers
        if not module_name.endswith("_provider"):
            continue

        importlib.import_module(f"{package_name}.{module_name}")