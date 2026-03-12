from rpgbot.services.dice_service import roll_dice


def fake_rng(a, b):
    return 10


def test_d20():

    result = roll_dice("1d20", rng=fake_rng)

    assert result.rolls == [10]
    assert result.total == 10

def test_advantage():

    values = [5, 17]

    def rng(a, b):

        return values.pop(0)

    result = roll_dice("2d20kh1", rng=rng)

    assert result.kept == [17]
    assert result.total == 17

def test_attribute_roll():

    values = [6, 5, 4, 1]

    def rng(a, b):

        return values.pop(0)

    result = roll_dice("4d6dl1", rng=rng)

    assert result.kept == [4,5,6]
    assert result.total == 15

def test_exploding_die():

    values = [6, 6, 2]

    def rng(a, b):
        return values.pop(0)

    result = roll_dice("1d6!", rng=rng)

    assert result.total == 14

def test_roll_detail():

    values = [6,5,4,1]

    def rng(a,b):
        return values.pop(0)

    result = roll_dice("4d6dl1", rng=rng)

    assert "kept" in result.detail
    assert result.total == 15