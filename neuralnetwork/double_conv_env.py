import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

def sort_by_distance(x, y, positions):
    def distance(position):
        px, py = position
        return (px - x) ** 2 + (py - y) ** 2

    return sorted(positions, key=distance)

class SnakeEnvLarge(gym.Env):

    def __init__(self, num_food=50):
        super(SnakeEnvLarge, self).__init__()
        self.grid_size = (50, 50)
        self.obersvation_size = (11, 11)
        self.low_res_factor = 5  # Adjust this factor as needed
        self.low_res_width = self.grid_size[0] // self.low_res_factor
        self.low_res_height = self.grid_size[1] // self.low_res_factor
        self.action_space = spaces.Discrete(3)  # Only 3 actions: rotate left (0) and rotate right (1) or do nothing

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0, high=255, shape=(self.obersvation_size[0], self.obersvation_size[1], 3), dtype=np.uint8
            ),
            "low_res_grid": spaces.Box(
                low=0, high=255, shape=(self.low_res_width, self.low_res_height, 3), dtype=np.uint8
            ),
            "additional_inputs": spaces.Box(
                low=0, high=1, shape=(4,), dtype=np.int64
            ),
        })

        self.keys = ["grid","low_res_grid",  "additional_inputs"]

        self.snake = None
        self.snake_set = None
        self.food = None
        self.num_food = num_food

        self.done = False
        self.direction = None
        self.score = 0
        self.step_cnt = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=random.randint(1000,1000000))
        self.snake = []
        self.snake_set = set()
        startX = random.randint(5, self.grid_size[0]-5)
        startY = random.randint(5, self.grid_size[1]-5)
        for x in range(3):
            pos = (startX - x, startY)
            self.snake.append(pos)
            self.snake_set.add(pos)

        self.direction = 1
        self.food = set()
        self._place_food()
        self.done = False
        self.score = 0
        self.step_cnt = 0

        return self._get_observation(), {}
    

    def direction_map(self):
        head = self.snake[0]
        return {
            0: (head[0] - 1, head[1]),  # Left
            1: (head[0], head[1] + 1),  # Down
            2: (head[0] + 1, head[1]),  # Right
            3: (head[0], head[1] - 1)   # Up
        }
    
    def is_killed(self, new_head):
        x_out = new_head[0] < 0 or new_head[0] >= self.grid_size[0]
        y_out = new_head[1] < 0 or new_head[1] >= self.grid_size[1]
        body_collision = new_head in self.snake_set

        return x_out or y_out or body_collision

    def step(self, action):
        if action not in [0, 1, 2]:
            print("action not in set!")
            exit()


        if action == 0: # rotate left
            if self.direction == 0: # left
                self.direction = 1
            elif self.direction == 1: # down
                self.direction = 2
            elif self.direction == 2: # right
                self.direction = 3
            elif self.direction == 3: # up
                self.direction = 0
        elif action == 1: # rotate right
            if self.direction == 0: # left
                self.direction = 3
            elif self.direction == 1: # down
                self.direction = 0
            elif self.direction == 2: # right
                self.direction = 1
            elif self.direction == 3: # up
                self.direction = 2
        elif action == 2:
            ... # do nothing
        else:
            print("unsupported action")
            exit()

        direction_map = self.direction_map()
        new_head = direction_map[self.direction]

        # Check for collisions
        killed = self.is_killed(new_head)

        reward = 0.1
        if killed:
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
                self.step_cnt += 1
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
            attempt = 0
            placed = False
            while not placed and attempt < max_attempts:
                pos = (random.randint(0, self.grid_size[0]-1),
                       random.randint(0, self.grid_size[1]-1))
                if pos not in existing:
                    self.food.add(pos)
                    existing.add(pos)
                    placed = True
                attempt += 1
            if not placed:
                print("could not place!")
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


        # DEEP EDIT?
        # Rotate grid based on snake's direction
        # if self.direction != 1:  # If not facing down, rotate grid
        #    grid = np.rot90(grid, k=self.direction - 1)

        rotation_map = {0: 3, 1: 0, 2: 1, 3: 2}  # Left: 270°, Down: 0°, Right: 90°, Up: 180°
        grid = np.rot90(grid, k=rotation_map[self.direction])

        # Determine food direction relative to snake's orientation
        if closest_food:
            dx = food_x - snake_x
            dy = food_y - snake_y

            if self.direction == 0:  # Facing left
                dx_rel, dy_rel = -dy, dx
            elif self.direction == 1:  # Facing down
                dx_rel, dy_rel = dx, dy
            elif self.direction == 2:  # Facing right
                dx_rel, dy_rel = dy, -dx
            elif self.direction == 3:  # Facing up
                dx_rel, dy_rel = -dx, -dy

            food_direction = np.array([
                int(dx_rel > 0 and dy_rel > 0),  # Bottom-right
                int(dx_rel < 0 and dy_rel > 0),  # Bottom-left
                int(dx_rel > 0 and dy_rel < 0),  # Top-right
                int(dx_rel < 0 and dy_rel < 0)   # Top-left
            ], dtype=np.int64)
        else:
            food_direction = np.zeros(4, dtype=np.int64)

        # Existing code to generate the local grid (grid) and food_direction...

        # Generate low-resolution grid
        low_res_grid = np.zeros((self.low_res_width, self.low_res_height, 3), dtype=np.uint8)
        for i in range(self.low_res_width):
            for j in range(self.low_res_height):
                x_start = i * self.low_res_factor
                x_end = x_start + self.low_res_factor
                y_start = j * self.low_res_factor
                y_end = y_start + self.low_res_factor

                has_head = False
                has_food = False
                has_body = False

                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        if (x, y) == self.snake[0]:
                            has_head = True
                        elif (x, y) in self.snake_set:
                            has_body = True
                        elif (x, y) in self.food:
                            has_food = True

                if has_head:
                    color = [0, 0, 255]  # Blue for head
                elif has_food:
                    color = [255, 0, 0]  # Red for food
                elif has_body:
                    color = [0, 255, 0]  # Green for body
                else:
                    color = [0, 0, 0]     # Black for empty

                low_res_grid[i, j] = color

        # Optional: Rotate low_res_grid to align with the snake's direction
        rotation_map = {0: 3, 1: 0, 2: 1, 3: 2}
        low_res_grid = np.rot90(low_res_grid, k=rotation_map[self.direction])

        return {
            "grid": grid,
            "low_res_grid": low_res_grid,
            "additional_inputs": food_direction
        }