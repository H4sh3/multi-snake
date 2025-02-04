import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from collections import deque
import torch
import torch.nn as nn

class SnakeEnvLarge(gym.Env):
    def __init__(self, num_food=5, frame_stack=4):
        super().__init__()
        self.grid_size = (20, 20)
        self.obersvation_size = (11, 11)
        self.frame_stack = frame_stack
        self.num_food = num_food
        
        # Observation space
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0, high=255, 
                shape=(self.frame_stack, self.obersvation_size[0], self.obersvation_size[1], 3),
                dtype=np.uint8
            ),
            "additional_inputs": spaces.Box(
                low=0, high=1, 
                shape=(4,),  # Single frame vector
                dtype=np.int64
            ),
        })

        # Action space
        self.action_space = spaces.Discrete(4)  # 0: left, 1: right, 2: noop, 3: boost
        
        # Game state
        self.snake = None
        self.snake_set = None
        self.food = None
        self.direction = None
        self.done = False
        self.score = 0
        self.step_cnt = 0
        self.grid_buffer = deque(maxlen=frame_stack)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize snake
        startX = random.randint(5, self.grid_size[0]-5)
        startY = random.randint(5, self.grid_size[1]-5)
        self.snake = [(startX - i, startY) for i in range(3)]
        self.snake_set = set(self.snake)
        
        # Game state
        self.direction = 1  # Initial direction: down
        self.food = set()
        self._place_food()
        self.done = False
        self.score = 0
        self.step_cnt = 0
        
        # Initialize grid buffer
        self.grid_buffer.clear()
        initial_grid = self._get_grid_observation()
        for _ in range(self.frame_stack):
            self.grid_buffer.append(initial_grid)
            
        return self._get_observation(), {}

    def _get_grid_observation(self):
        grid = np.zeros((self.obersvation_size[0], self.obersvation_size[1], 3), dtype=np.uint8)
        half = self.obersvation_size[0] // 2
        head_x, head_y = self.snake[0]
        
        # Populate grid
        for dx in range(-half, half+1):
            for dy in range(-half, half+1):
                grid_x = half + dx
                grid_y = half + dy
                world_x = head_x + dx
                world_y = head_y + dy
                
                # Check boundaries
                if (world_x < 0 or world_x >= self.grid_size[0] or
                    world_y < 0 or world_y >= self.grid_size[1]):
                    grid[grid_x, grid_y] = [255, 255, 255]  # Wall
                elif (world_x, world_y) in self.snake_set:
                    grid[grid_x, grid_y] = [0, 255, 0]      # Snake body
                elif (world_x, world_y) == self.snake[0]:
                    grid[grid_x, grid_y] = [0, 0, 255]      # Snake head
                elif (world_x, world_y) in self.food:
                    grid[grid_x, grid_y] = [255, 0, 0]      # Food
        
        # Rotate based on direction
        rotation_map = {0: 3, 1: 0, 2: 1, 3: 2}
        grid = np.rot90(grid, k=rotation_map[self.direction])
        return grid

    def _get_additional_inputs(self):
        if not self.food:
            return np.zeros(4, dtype=np.int64)
            
        head = self.snake[0]
        closest = min(self.food, key=lambda p: (p[0]-head[0])**2 + (p[1]-head[1])**2)
        dx = closest[0] - head[0]
        dy = closest[1] - head[1]
        
        # Rotate coordinates based on snake direction
        if self.direction == 0:   # Left
            dx, dy = dy, -dx
        elif self.direction == 2: # Right
            dx, dy = -dy, dx
        elif self.direction == 3: # Up
            dx, dy = -dx, -dy
            
        return np.array([
            int(dx > 0 and dy > 0),   # Right-down
            int(dx < 0 and dy > 0),   # Left-down
            int(dx > 0 and dy < 0),   # Right-up
            int(dx < 0 and dy < 0)    # Left-up
        ], dtype=np.int64)

    def step(self, action):
        reward = 0
        original_direction = self.direction
        
        # Handle direction change
        if action == 0:  # Left
            self.direction = (self.direction - 1) % 4
        elif action == 1:  # Right
            self.direction = (self.direction + 1) % 4
            
        # Handle boost action (2 steps)
        if action == 3:
            reward += self._move(original_direction)
            if not self.done:
                reward += self._move(original_direction)
        else:
            reward += self._move(self.direction)
        
        # Update grid buffer
        self.grid_buffer.append(self._get_grid_observation())
        
        return self._get_observation(), reward, self.done, False, {}

    def _move(self, direction):
        # Calculate new head position
        head = self.snake[0]
        dir_map = {
            0: (head[0]-1, head[1]),  # Left
            1: (head[0], head[1]+1),  # Down
            2: (head[0]+1, head[1]),  # Right
            3: (head[0], head[1]-1)   # Up
        }
        new_head = dir_map[direction]
        
        # Check collision
        if (new_head in self.snake_set or
            new_head[0] < 0 or new_head[0] >= self.grid_size[0] or
            new_head[1] < 0 or new_head[1] >= self.grid_size[1]):
            self.done = True
            return -1
        
        # Move snake
        self.snake.insert(0, new_head)
        self.snake_set.add(new_head)
        
        # Check food
        if new_head in self.food:
            self.food.remove(new_head)
            self._place_food()
            self.score += 1
            return 1
        else:
            # Remove tail
            tail = self.snake.pop()
            self.snake_set.remove(tail)
            return 0.1

    def _place_food(self):
        existing = self.snake_set.union(self.food)
        while len(self.food) < self.num_food:
            pos = (random.randint(0, self.grid_size[0]-1), 
                   random.randint(0, self.grid_size[1]-1))
            if pos not in existing:
                self.food.add(pos)
                existing.add(pos)

    def _get_observation(self):
        return {
            "grid": np.array(self.grid_buffer),
            "additional_inputs": self._get_additional_inputs()
        }

class SnakeNet3D(nn.Module):
    def __init__(self, frame_stack=4):
        super().__init__()
        # 3D CNN for spatial-temporal patterns
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Flatten()
        )
        
        # Vector processor
        self.vec_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU()
        )
        
        # Combined network
        self.fc = nn.Sequential(
            nn.Linear(64*4*2*2 + 64, 512),  # Adjust based on conv output
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        # Process grid (batch, frames, h, w, channels) -> (batch, channels, frames, h, w)
        grid = x["grid"].float().permute(0, 4, 1, 2, 3)
        conv_out = self.conv3d(grid)
        
        # Process vector
        vec = x["additional_inputs"].float()
        vec_out = self.vec_net(vec)
        
        # Combine
        combined = torch.cat([conv_out, vec_out], dim=1)
        return self.fc(combined)

# Example usage
if __name__ == "__main__":
    env = SnakeEnvLarge(frame_stack=4)
    model = SnakeNet3D()
    
    obs, _ = env.reset()
    print("Observation shapes:")
    print(f"Grid: {obs['grid'].shape}, Vector: {obs['additional_inputs'].shape}")
    
    # Convert to tensor
    grid_tensor = torch.tensor(obs["grid"]).unsqueeze(0)
    vec_tensor = torch.tensor(obs["additional_inputs"]).unsqueeze(0)
    
    # Test forward pass
    with torch.no_grad():
        output = model({"grid": grid_tensor, "additional_inputs": vec_tensor})
    print(f"Model output shape: {output.shape}")