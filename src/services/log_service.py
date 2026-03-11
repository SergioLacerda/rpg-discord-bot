import os
from datetime import datetime
from src.config import LOG_PATH


def write_log(text: str):

    date = datetime.now().strftime("%Y-%m-%d")

    file = os.path.join(LOG_PATH, f"session_{date}.md")

    with open(file, "a") as f:

        f.write(text + "\n")

    return file