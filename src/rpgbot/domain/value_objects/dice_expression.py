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