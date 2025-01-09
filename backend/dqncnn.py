import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import numpy as np
import config

class SnakeCNN(nn.Module):
    def __init__(self, height, width, num_actions, num_channels):
        super(SnakeCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        conv_out_size = (height) * (width) * 64 # Adjusted for pooling

        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, num_actions)
        
        # Additional layers
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm2d(64)

    def forward(self, x):
        # Convolutional layers
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = self.batch_norm(x)                  # Batch normalization
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)                     # Dropout
        x = self.fc2(x)
        return x

class SnakeCNNNEW(nn.Module):
    def __init__(self, num_actions):
        super(SnakeCNNNEW, self).__init__()
        
        # Conv3d: (channels, depth, height, width)
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 3 * 6 * 6, out_features=128)  # 64 channels * 3 frames * 6x6 grid
        self.fc2 = nn.Linear(in_features=128, out_features=num_actions)
        
    def forward(self, x):
        # Input: (batch_size, channels=3, frames=3, height=6, width=6)
        
        # Conv3d layers with ReLU activation
        x = F.relu(self.conv1(x))  # Output shape: (batch_size, 16, 3, 6, 6)
        x = F.relu(self.conv2(x))  # Output shape: (batch_size, 32, 3, 6, 6)
        x = F.relu(self.conv3(x))  # Output shape: (batch_size, 64, 3, 6, 6)
        
        # Flatten the output
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64 * 3 * 6 * 6)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class SnakeCNN3D(nn.Module):
    def __init__(self, depth, height, width, num_actions, num_channels):
        super(SnakeCNN3D, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Calculate the output size after convolutions and pooling
        conv_out_size = (depth // 2) * (height // 2) * (width // 2) * 64  # Account for pooling reducing dimensions by half
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, num_actions)
        
        # Additional layers
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm3d(64)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        
        x = self.conv2(x)
        #x = self.batch_norm(x)
        x = nn.ReLU()(x)
        
        x = self.pool(x)  # Reduce spatial dimensions
        
        x = torch.flatten(x, start_dim=1)  # Flatten for fully connected layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x
    

class CNNDQNAgent:
    def __init__(self, depth:int = 3 ,height: int = 15, width: int = 15, num_actions: int = 4, learning_rate: float = 1e-4, gamma: float = 0.99, num_channels: int = 3):
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = config.device

        # Neural network
        self.policy_net = SnakeCNN3D(depth, height, width, num_actions, num_channels).to(self.device)
        self.target_net = SnakeCNN3D(depth, height, width, num_actions, num_channels).to(self.device)

        #self.policy_net = SnakeCNNNEW(num_actions).to(self.device)
        #self.target_net = SnakeCNNNEW(num_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay memory
        self.memory = deque(maxlen=10000)
        self.pos_memory = deque(maxlen=10000)
        self.normal_memory = deque(maxlen=10000)
        self.short_memory = deque(maxlen=10000)

        # Exploration parameters
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        self.max_reward = 0

        self.frames = deque(maxlen=3)

    def select_action(self, state_tensor) -> int:
        """Select an action based on epsilon-greedy policy."""
        # state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)  # Explore
        return q_values.argmax().item()  # Exploit

    def store_short_memory(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.short_memory.append((state, action, reward, next_state, done))

    def process_short_memory(self):
        """Process short-term memory for replay."""
        self.memory += self.short_memory
        self.short_memory.clear()
        return
    

        slices = []
        sublist = []
        for i, item in enumerate(self.short_memory):
            sublist.append(item)
            if item[2] in [1, -1]:
                slices.append(sublist)
                sublist = []
            else:
                sublist.append(item)

        reward_sum = 0

        for memory_slice in slices:
            for i, experience in enumerate(reversed(memory_slice)):
                state, action, reward, next_state, done = experience
                if reward == 1 or reward == -1:
                    tmp_memories = []
                    self.memory.append(experience)
                    for i, prev_experience in enumerate(reversed(memory_slice)):
                        if i == 0: continue  # Already added
                        parts = 1 / len(memory_slice)
                        new_reward = reward * parts * i
                        reward_sum += new_reward
                        prev_state, prev_action, prev_reward, prev_next_state, prev_done = prev_experience
                        modified_experience = (prev_state, prev_action, new_reward, prev_next_state, prev_done)
                        tmp_memories.append(modified_experience)

                    tmp_memories.reverse()
                    self.memory += tmp_memories

        self.short_memory.clear()

    def train_step(self, batch_size: int):
        """Train the network using a batch of experiences."""
        if len(self.memory) < batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Current Q-values
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss and optimization
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update_target_network(self):
        """Update the target network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self, episode, max_episode):
        self.epsilon = 0.9 - (episode / (max_episode/2)) * (0.9 - 0.1)
        self.epsilon = max(self.epsilon, 0.1)  # Ensure epsilon does not go below 0.1

    def save_model(self, filename: str):
        torch.save(self.policy_net.state_dict(), filename)
