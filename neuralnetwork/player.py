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
        self.init_steps_to_eat = 25
        self.reset_ste()

    def reset_ste(self):
        self.steps_to_eat = self.init_steps_to_eat

    def turn(self, new_direction: Direction):
        if not self.alive:
            return

        current = self.direction
        # should not turn on the spot
        if new_direction.value[0] + current.value[0] == 0 and new_direction.value[1] + current.value[1] == 0:
            return

        self.direction = new_direction