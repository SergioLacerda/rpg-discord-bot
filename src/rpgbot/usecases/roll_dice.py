from typing import Callable
import random

from rpgbot.domain.dice.parser import parse_dice
from rpgbot.domain.dice.roller import roll
from rpgbot.domain.entities.dice_result import DiceResult


def roll_dice(expr: str, rng: Callable[[int, int], int] | None = None) -> DiceResult:
    if rng is None:
        rng = random.randint

    dice = parse_dice(expr)

    result = roll(dice, rng=rng)

    return result