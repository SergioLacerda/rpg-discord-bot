
from pathlib import Path

class DocumentLoader:

    def __init__(self, campaign_dir):
        self.campaign_dir = campaign_dir

    def discover(self):

        docs = []

        for file in self.campaign_dir.glob("**/*.md"):

            docs.append({
                "source": str(file),
                "text": file.read_text(encoding="utf-8"),
                "mtime": file.stat().st_mtime
            })

        return docs