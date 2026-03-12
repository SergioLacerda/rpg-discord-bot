from datetime import datetime
import os
from rpgbot.config import LOG_PATH


def write_log(text: str):

    LOG_PATH.mkdir(parents=True, exist_ok=True)

    date = datetime.now().strftime("%Y-%m-%d")

    file = os.path.join(LOG_PATH, f"session_{date}.md")

    with open(file, "a") as f:
        f.write(text + "\n")

    return file