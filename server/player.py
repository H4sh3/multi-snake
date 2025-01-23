from enum import Enum

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class Player:
    def __init__(self, sid, grid_size):
        self.sid = sid
        self.alive = True
        self.ready = False
        self.direction = Direction.RIGHT
        start_x = grid_size[0] // 2
        start_y = grid_size[1] // 2
        self.trail = [(start_x, start_y)]

    def turn(self, direction):
        if direction in Direction:
            # Prevent 180-degree turn
            if (self.direction.value[0] + direction.value[0] != 0 or
                    self.direction.value[1] + direction.value[1] != 0):
                self.direction = direction

    def move(self):
        head_x, head_y = self.trail[0]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)
        self.trail.insert(0, new_head)

    def grow(self):
        # No need to pop the tail if growing
        pass

    def to_dict(self):
        return {
            "sid": self.sid,
            "trail": self.trail,
            "alive": self.alive,
            "ready": self.ready,
        }
