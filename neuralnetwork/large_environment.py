import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class SnakeEnvLarge(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = (50, 50)
        self.obersvation_size = (10, 10)
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Dict({
            "grid":spaces.Box(
                low=0, high=255, shape=(self.obersvation_size[0], self.obersvation_size[1], 3), dtype=np.uint8
            ),
            "food_direction":spaces.Discrete(4)
        })

        self.snake = None
        self.food = None
        self.done = False
        self.direction = None
        self.score = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = []
        startX = random.randint(5,self.grid_size[0]-5)
        startY = random.randint(5,self.grid_size[1]-5)
        for x in range(3):
            self.snake.append((startX - x, startY))

        self.direction = 1
        self.food = self._place_food()
        self.done = False
        self.score = 0

        return self._get_observation(), {}

    def step(self, action):
        if action in [0, 1, 2, 3]:
            self.direction = action

        head = self.snake[0]
        if self.direction == 0:
            new_head = (head[0] - 1, head[1])
        elif self.direction == 1:
            new_head = (head[0], head[1] + 1)
        elif self.direction == 2:
            new_head = (head[0] + 1, head[1])
        elif self.direction == 3:
            new_head = (head[0], head[1] - 1)

        if (
            new_head[0] < 0 or new_head[0] >= self.grid_size[0]
            or new_head[1] < 0 or new_head[1] >= self.grid_size[1]
            or new_head in self.snake
        ):
            self.done = True
            reward = -1
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.food = self._place_food()
                reward = 1
                self.score += 1
                if not self.food:
                    self.done = True
            else:
                self.snake.pop()
                reward = 0

        return self._get_observation(), reward, self.done, False, {}


    def _place_food(self):
        # Todo: optimize with random spawn - running out of spawn positions is no problem in the large environment
        all_positions = set(
            (x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])
        )
        empty_positions = list(all_positions - set(self.snake))
        if not empty_positions:
            return None
        return random.choice(empty_positions)

    def _get_observation(self):
        grid = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)
        for x, y in self.snake:
            grid[x, y] = [0, 255, 0]
        grid[self.snake[0][0], self.snake[0][1]] = [0, 0, 255]
        if self.food:
            food_x, food_y = self.food
            grid[food_x, food_y] = [255, 0, 0]

        snake_x = self.snake[0][0]
        snake_y = self.snake[0][1]
        food_x = self.food[0]
        food_y = self.food[1]
        food_direction = None

        if food_x > snake_x and food_y > snake_y:
            food_direction = [1,0,0,0]
        elif food_x < snake_x and food_y > snake_y:
            food_direction = [0,1,0,0]
        elif food_x > snake_x and food_y < snake_y:
            food_direction = [0,0,1,0]
        elif food_x < snake_x and food_y < snake_y:
            food_direction = [0,0,0,1]

        return (grid, food_direction)
