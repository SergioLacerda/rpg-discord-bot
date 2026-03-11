import re
from dataclasses import dataclass


@dataclass
class DiceExpression:

    num_dice: int
    sides: int

    explode: bool = False

    keep_high: int | None = None
    keep_low: int | None = None
    drop_high: int | None = None
    drop_low: int | None = None

    modifier: int = 0


pattern = re.compile(
    r"(?P<num>\d+)d(?P<sides>\d+)"
    r"(?P<explode>!)?"
    r"(?:(?P<keep>(kh|kl|dh|dl))(?P<count>\d+))?"
    r"(?P<mod>[+-]\d+)?"
)


def parse_dice(expr: str) -> DiceExpression:

    match = pattern.fullmatch(expr)

    if not match:
        raise ValueError("Invalid dice expression")

    num = int(match.group("num"))
    sides = int(match.group("sides"))

    explode = bool(match.group("explode"))

    mod = match.group("mod")
    modifier = int(mod) if mod else 0

    keep = match.group("keep")
    count = match.group("count")

    exp = DiceExpression(
        num_dice=num,
        sides=sides,
        explode=explode,
        modifier=modifier
    )

    if keep:

        count = int(count)

        if keep == "kh":
            exp.keep_high = count

        elif keep == "kl":
            exp.keep_low = count

        elif keep == "dh":
            exp.drop_high = count

        elif keep == "dl":
            exp.drop_low = count

    return exp