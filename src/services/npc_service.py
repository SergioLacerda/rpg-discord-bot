import random

names = [
    "Arkan",
    "Velra",
    "Doran",
    "Selith",
    "Kara",
    "Nox"
]

traits = [
    "frio",
    "calculista",
    "nervoso",
    "irônico",
    "silencioso"
]


def generate_npc(desc: str) -> dict:

    return {

        "name": random.choice(names),
        "description": desc,
        "trait": random.choice(traits)
    }