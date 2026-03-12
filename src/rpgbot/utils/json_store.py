import json
from pathlib import Path


def load_json(path, default):

    if path.exists():
        return json.loads(path.read_text())

    return default


def save_json(path, data):

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(json.dumps(data, indent=2))
