from typing import List, Tuple
from enum import Enum


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class Player:
    def __init__(self, x: int, y: int, direction: Direction, color: str):
        self.x = x
        self.y = y
        self.direction = direction
        self.color = color
        self.trail: List[Tuple[int, int]] = [(x, y)]
        self.alive = True
        self.ready = False
        self.sid = None
        self.wins = 0
