from rpgbot.domain.entities.dice_result import DiceResult
from rpgbot.domain.dice.parser import DiceExpression


def roll(expr: DiceExpression, rng):

    rolls = []

    for _ in range(expr.num_dice):

        value = rng(1, expr.sides)

        rolls.append(value)

        if expr.explode:
            while value == expr.sides:
                value = rng(1, expr.sides)
                rolls.append(value)

    kept = rolls[:]

    if expr.keep_high:
        kept = sorted(rolls, reverse=True)[:expr.keep_high]

    if expr.keep_low:
        kept = sorted(rolls)[:expr.keep_low]

    if expr.drop_high:
        kept = sorted(rolls)[:-expr.drop_high]

    if expr.drop_low:
        kept = sorted(rolls)[expr.drop_low:]

    total = sum(kept) + expr.modifier

    detail = (
        f"🎲 {expr.num_dice}d{expr.sides}\n"
        f"rolls : {rolls}\n"
        f"kept  : {kept}\n"
        f"total : {total}"
    )

    return DiceResult(
        expression="",
        rolls=rolls,
        kept=kept,
        modifier=expr.modifier,
        total=total,
        detail=detail,
    )