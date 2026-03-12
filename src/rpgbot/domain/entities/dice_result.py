from dataclasses import dataclass


@dataclass
class DiceResult:

    expression: str
    rolls: list[int]
    kept: list[int]
    modifier: int
    total: int
    detail: str