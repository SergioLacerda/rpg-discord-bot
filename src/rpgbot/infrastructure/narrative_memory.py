import json
from pathlib import Path

MEMORY_PATH = Path("campaign/memory/narrative_summary.json")


class NarrativeMemory:

    def __init__(self):

        self.world_facts = []
        self.scene_state = []
        self.recent_events = []
        self.summary = ""

        if MEMORY_PATH.exists():
            try:
                self.summary = json.loads(
                    MEMORY_PATH.read_text(encoding="utf-8")
                ).get("summary", "")
            except Exception:
                self.summary = ""

    def get(self) -> str:

        parts = []

        if self.scene_state:
            parts.append("=== CURRENT SCENE ===")
            parts.extend(self.scene_state)

        if self.recent_events:
            parts.append("=== RECENT EVENTS ===")
            parts.extend(self.recent_events)

        return "\n".join(parts)

    def update(self, event: str):

        if not event:
            return

        self.recent_events.append(event)

        # mantém apenas últimos eventos
        self.recent_events = self.recent_events[-5:]

        # heurística simples para estado da cena
        keywords = ["agora", "neste momento", "está", "encontra-se"]

        if any(k in event.lower() for k in keywords):
            self.scene_state.append(event)
            self.scene_state = self.scene_state[-3:]

    def _persist(self):

        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

        MEMORY_PATH.write_text(
            json.dumps({"summary": self.summary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


memory = NarrativeMemory()