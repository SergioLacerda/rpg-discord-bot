from rpgbot.usecases.generate_npc import generate_npc


def test_npc_generation():

    npc = generate_npc("mercador")

    assert "name" in npc
    assert npc["description"] == "mercador"