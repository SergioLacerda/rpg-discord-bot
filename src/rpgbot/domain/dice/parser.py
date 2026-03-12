import re
from dataclasses import dataclass
from rpgbot.domain.value_objects.dice_expression import DiceExpression

_PATTERN = re.compile(
    r"(?P<num>\d+)d(?P<sides>\d+)"
    r"(?P<explode>!)?"
    r"(?:(?P<keep>(kh|kl|dh|dl))(?P<count>\d+))?"
    r"(?P<mod>[+-]\d+)?"
)


def parse_dice(expr: str) -> DiceExpression:
    expr = expr.strip().lower()

    match = _PATTERN.fullmatch(expr)

    if not match:
        raise ValueError("Invalid dice expression")

    num = int(match.group("num"))
    sides = int(match.group("sides"))

    explode = bool(match.group("explode"))

    mod = match.group("mod")
    modifier = int(mod) if mod else 0

    keep = match.group("keep")
    count = match.group("count")

    dice = DiceExpression(
        num_dice=num,
        sides=sides,
        explode=explode,
        modifier=modifier,
    )

    if keep:

        count = int(count)

        if keep == "kh":
            dice.keep_high = count

        elif keep == "kl":
            dice.keep_low = count

        elif keep == "dh":
            dice.drop_high = count

        elif keep == "dl":
            dice.drop_low = count

    return dice