import random
import logging

from rpgbot.models.dice_result import DiceResult
from rpgbot.services.dice_parser import parse_dice


logger = logging.getLogger(__name__)


def roll_exploding_die(sides: int, rng):

    rolls = []

    while True:

        roll = rng(1, sides)

        rolls.append(roll)

        if roll != sides:
            break

        logger.debug(f"Dado explodiu | sides={sides}")

    return rolls


def roll_pool(parsed, rng):

    rolls = []

    for _ in range(parsed.num_dice):

        if parsed.explode:

            sub_rolls = roll_exploding_die(parsed.sides, rng)

            rolls.extend(sub_rolls)

        else:

            rolls.append(rng(1, parsed.sides))

    return rolls


def apply_keep_drop(rolls, parsed):

    kept = rolls.copy()

    if parsed.keep_high:

        kept = sorted(rolls, reverse=True)[: parsed.keep_high]

    elif parsed.keep_low:

        kept = sorted(rolls)[: parsed.keep_low]

    elif parsed.drop_high:

        kept = sorted(rolls)[:-parsed.drop_high]

    elif parsed.drop_low:

        kept = sorted(rolls)[parsed.drop_low:]

    return kept


def roll_dice(expr: str, rng=random.randint) -> DiceResult:

    parsed = parse_dice(expr)

    logger.debug(f"Parsed dice | expr={expr} | parsed={parsed}")

    rolls = roll_pool(parsed, rng)

    kept = apply_keep_drop(rolls, parsed)

    total = sum(kept) + parsed.modifier

    detail = build_roll_detail(expr, rolls, kept, parsed.modifier)

    logger.info(detail)

    return DiceResult(
        expr,
        rolls,
        kept,
        parsed.modifier,
        total,
        detail
    )

def build_roll_detail(expr, rolls, kept, modifier):

    roll_str = ", ".join(map(str, rolls))
    kept_str = ", ".join(map(str, kept))

    base = f"{expr} → rolls [{roll_str}]"

    if kept != rolls:
        base += f" → kept [{kept_str}]"

    if modifier:
        base += f" {modifier:+d}"

    base += f" = {sum(kept) + modifier}"

    return base