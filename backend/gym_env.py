import numpy as np
from typing import Dict, Tuple, Any
from player import Player, Direction
from helpers import get_inputs_cnn
import random

from stable_baselines3.common.vec_env import VecEnv

import gymnasium as gym
from gymnasium import spaces

class GameEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width: int = 15, height: int = 15):
        super(GameEnvironment, self).__init__()
        self.width = width
        self.height = height
        self.player: Player = Player(random.randint(0,width),random.randint(0,height),random.choice([Direction.UP,Direction.DOWN,Direction.LEFT,Direction.RIGHT]),"FF00FF")
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.food: Tuple[int, int] = 0, 0
        self.spawn_food()
        self.game_active = False

        # Define action and observation spaces
        # Actions: {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.action_space = spaces.Discrete(4)

        # Observations: grid state flattened into a 1D array
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, width, height), dtype=np.float32
        )

    def spawn_food(self, x_spawn: int = None, y_spawn: int = None):
        while True:
            x = x_spawn if x_spawn is not None else np.random.randint(0, self.width)
            y = y_spawn if y_spawn is not None else np.random.randint(0, self.height)
            if self.grid[x][y] is None:
                self.grid[x][y] = "FOOD"
                self.food = x, y
                return

    def reset(self, seed: int | None = None,):
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        positions = [
            (5, 5, Direction.RIGHT, "#FF0000"),
            (self.width - 5, self.height - 5, Direction.LEFT, "#0000FF"),
            (5, self.height - 5, Direction.RIGHT, "#00FF00"),
            (self.width - 5, 5, Direction.LEFT, "#FFFF00")
        ]
        self.player.alive = True
        self.player.ready = False
        self.player.steps_to_eat = 25
        x, y, direction, color = positions[0]
        self.player.x = x
        self.player.y = y
        self.player.direction = direction
        self.player.color = color
        self.player.trail = [(x, y)] * 5
        self.grid[y][x] = color
        self.spawn_food()
        self.game_active = True

        # Return initial observation
        obs = self._grid_to_observation()
        return (obs, {})

    def step(self, action: int):
        # Map action to direction
        direction_map = {
            0: Direction.UP,
            1: Direction.DOWN,
            2: Direction.LEFT,
            3: Direction.RIGHT
        }
        self.player.turn(direction_map[action])  # Assume single-player for now

        reward = 0
        done = False
        info = {}

        if not self.player.alive:
            done = True
            return (self._grid_to_observation(), -1, done, False, info)


        dx, dy = self.player.direction.value
        new_x = self.player.x + dx
        new_y = self.player.y + dy

        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
            done = True
            reward = -1
            return (self._grid_to_observation(), reward, done, False, info)

        if self.grid[new_y][new_x] is not None and self.grid[new_y][new_x] != 'FOOD':
            done = True
            reward = -1
            return (self._grid_to_observation(), reward, done, False, info)

        if (new_x, new_y) == self.food:
            reward = 1
            self.player.trail.append((new_x, new_y))
            self.spawn_food()
        else:
            tail_end = self.player.trail.pop(0)
            self.grid[tail_end[1]][tail_end[0]] = None

        self.player.x = new_x
        self.player.y = new_y
        self.player.trail.append((new_x, new_y))
        self.grid[new_y][new_x] = self.player.color

        return (self._grid_to_observation(), reward, done, False, info)

    def render(self, mode='human'):
        visual_grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        food_x, food_y = self.food
        visual_grid[food_y][food_x] = 'F'

        if self.player.alive:
            for idx, (x, y) in enumerate(self.player.trail):
                if idx == len(self.player.trail) - 1:
                    visual_grid[y][x] = 'X'
                else:
                    visual_grid[y][x] = 'x'

        print("\n" + "-" * (self.width + 2))
        for row in visual_grid:
            print("|" + "".join(row) + "|")
        print("-" * (self.width + 2))

    def close(self):
        pass

    def _grid_to_observation(self):
        obs = get_inputs_cnn(grid=self.grid, food=self.food, player=self.player)
        #print(obs)
        #print(obs.shape)
        #print(type(obs))
        return obs
