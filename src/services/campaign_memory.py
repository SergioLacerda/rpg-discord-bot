from pathlib import Path

CAMPAIGN_DIR = Path("campaign")


def load_campaign_context():

    text = []

    for file in CAMPAIGN_DIR.rglob("*.md"):

        with open(file) as f:

            text.append(f.read())

    return "\n\n".join(text)