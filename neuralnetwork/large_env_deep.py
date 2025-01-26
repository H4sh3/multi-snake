import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import math

def sort_by_distance(x, y, positions):
    def distance(position):
        px, py = position
        return (px - x) ** 2 + (py - y) ** 2  # Avoid sqrt for performance

    return sorted(positions, key=distance)

class SnakeEnvLarge(gym.Env):
    def __init__(self):
        super(SnakeEnvLarge, self).__init__()
        self.grid_size = (50, 50)
        self.obersvation_size = (11, 11)
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0, high=255, shape=(self.obersvation_size[0], self.obersvation_size[1], 3), dtype=np.uint8
            ),
            "food_direction": spaces.Box(
                low=0, high=1, shape=(4,), dtype=np.int64
            )
        })

        self.keys = ["grid", "food_direction"]

        self.snake = None
        self.snake_set = None  # For O(1) lookups
        self.food = None        # Now a set for O(1) lookups
        self.num_food = 10

        self.done = False
        self.direction = None
        self.score = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = []
        self.snake_set = set()
        startX = random.randint(5, self.grid_size[0]-5)
        startY = random.randint(5, self.grid_size[1]-5)
        for x in range(20):
            pos = (startX - x, startY)
            self.snake.append(pos)
            self.snake_set.add(pos)

        self.direction = 1
        self.food = set()
        self._place_food()
        self.done = False
        self.score = 0

        return self._get_observation(), {}

    def step(self, action):
        illegal_action = (self.direction == 0 and action == 2) or \
                         (self.direction == 2 and action == 0) or \
                         (self.direction == 1 and action == 3) or \
                         (self.direction == 3 and action == 1)

        reward = -1 if illegal_action else 0

        if not illegal_action and action in [0, 1, 2, 3]:
            self.direction = action

        head = self.snake[0]
        direction_map = {
            0: (head[0] - 1, head[1]),  # Left
            1: (head[0], head[1] + 1),  # Down
            2: (head[0] + 1, head[1]),  # Right
            3: (head[0], head[1] - 1)   # Up
        }
        new_head = direction_map[self.direction]

        # Check for collisions
        x_out = new_head[0] < 0 or new_head[0] >= self.grid_size[0]
        y_out = new_head[1] < 0 or new_head[1] >= self.grid_size[1]
        body_collision = new_head in self.snake_set

        if x_out or y_out or body_collision:
            self.done = True
            reward = -1
        else:
            self.snake.insert(0, new_head)
            self.snake_set.add(new_head)
            if new_head in self.food:
                self.food.remove(new_head)
                self._place_food()
                reward = 1
                self.score += 1
                if not self.food:
                    self.done = True
            else:
                # Remove tail
                tail = self.snake.pop()
                self.snake_set.remove(tail)

        return self._get_observation(), reward, self.done, False, {}

    def _place_food(self):
        existing = self.snake_set.union(self.food)
        grid_area = self.grid_size[0] * self.grid_size[1]
        if len(existing) >= grid_area:
            return  # No space left

        max_attempts = 1000
        while len(self.food) < self.num_food:
            attempts = 0
            placed = False
            while not placed and attempts < max_attempts:
                pos = (random.randint(0, self.grid_size[0]-1),
                       random.randint(0, self.grid_size[1]-1))
                if pos not in existing:
                    self.food.add(pos)
                    existing.add(pos)
                    placed = True
                attempts += 1
            if not placed:
                break  # Could not place food

    def _get_observation(self):
        obs = self.obersvation_size[0]
        grid = np.zeros((obs, obs, 3), dtype=np.uint8)
        half = obs // 2
        center = self.snake[0]
        snake_x, snake_y = center

        # Find closest food efficiently
        closest_food = None
        if self.food:
            closest_food = min(self.food, key=lambda p: (p[0]-snake_x)**2 + (p[1]-snake_y)**2)
            food_x, food_y = closest_food
        else:
            food_x, food_y = -1, -1  # Out of bounds

        # Precompute relative positions
        for xOff in range(-half, half + 1):
            for yOff in range(-half, half + 1):
                grid_x = half + xOff
                grid_y = half + yOff
                world_x = center[0] + xOff
                world_y = center[1] + yOff

                # Check boundaries
                if (world_x < 0 or world_x >= self.grid_size[0] or
                    world_y < 0 or world_y >= self.grid_size[1]):
                    grid[grid_x, grid_y] = [255, 255, 255]  # Boundary
                elif (world_x, world_y) in self.snake_set:
                    grid[grid_x, grid_y] = [0, 255, 0]      # Snake body
                elif closest_food and (world_x, world_y) == (food_x, food_y):
                    grid[grid_x, grid_y] = [255, 0, 0]      # Closest food

        # Snake head at center
        grid[half, half] = [0, 0, 255]

        # Determine food direction
        if closest_food:
            dx = food_x - snake_x
            dy = food_y - snake_y
            food_direction = np.array([
                int(dx > 0 and dy > 0),  # Bottom-right
                int(dx < 0 and dy > 0),  # Bottom-left
                int(dx > 0 and dy < 0),  # Top-right
                int(dx < 0 and dy < 0)   # Top-left
            ], dtype=np.int64)
        else:
            food_direction = np.zeros(4, dtype=np.int64)

        return {
            "grid": grid,
            "food_direction": food_direction
        }