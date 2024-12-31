import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size: int, num_actions: int):
        super(DQN, self).__init__()
        # Use a fully connected network instead of convolutions
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),  # Input layer
            nn.ReLU(),
            nn.Linear(32, 16),          # Hidden layer
            nn.ReLU(),
            nn.Linear(16, num_actions)   # Output layer
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, input_size: int, num_actions: int, learning_rate: float = 1e-3, gamma: float = 0.99):
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural network
        self.policy_net = DQN(input_size, num_actions).to(self.device)
        self.target_net = DQN(input_size, num_actions).to(self.device)
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

    def select_action(self, state: np.ndarray) -> int:
        """Select an action based on epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)  # Explore
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()  # Exploit

    def store_short_memory(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.short_memory.append((state, action, reward, next_state, done))

    def process_short_memory(self):
        if any([x[2] == 1 for x in self.short_memory]):
            # print("Found a reward of 1. Storing experiences with diminishing rewards.")
            acc_reward = 0
            for i, experience in enumerate(reversed(self.short_memory)):
                state, action, reward, next_state, done = experience
                acc_reward += reward
                if reward == 1:
                    tmp_memories = []
                    # Add the current experience to memory
                    self.memory.append(experience)
                    # Calculate diminishing rewards for preceding experiences
                    discount_factor = 0.9  # Example discount factor
                    for j in range(i - 1, -1, -1):
                        prev_experience = self.short_memory[j]
                        prev_state, prev_action, prev_reward, prev_next_state, prev_done = prev_experience
                        if prev_reward == 0:  # Only modify experiences with no reward initially
                            new_reward = discount_factor ** (i - j) * reward
                            modified_experience = (prev_state, prev_action, new_reward, prev_next_state, prev_done)
                            tmp_memories.append(modified_experience)

                    # break  # Stop processing once we find the first reward of 1
            tmp_memories.reverse()
            self.memory += tmp_memories

            #if acc_reward > self.max_reward:
            #    self.max_reward = acc_reward
            #    self.pos_memory += tmp_memories

        else:
            if random.random() > 0.9: # only store certain part of bad memories
                # print("No reward of 1 found. Storing experiences without modification.")
                tmp_memories = [list(x) for x in self.short_memory]
                for x in tmp_memories:
                    x[2] = 0

                self.memory += tmp_memories

        # Clear short_memory after processing
        self.short_memory.clear()

    def train_step2(self, batch_size: int):
        """Train the network using a batch of experiences."""
        #print(len(self.memory))

        len_dq1 = len(self.pos_memory)
        len_dq2 = len(self.normal_memory)
        
        pos_elements = int(len_dq1 * 0.8)
        norm_elements = int(len_dq2 * 0.2)

        from collections import deque
        import random

        dq_list = list(self.pos_memory)
        random.shuffle(dq_list)
        self.pos_memory = deque(dq_list)

        dq_list = list(self.normal_memory)
        random.shuffle(dq_list)
        self.normal_memory = deque(dq_list)

        # Extract the required number of elements from each deque
        new_deque = deque(list(self.pos_memory)[:pos_elements] + list(self.normal_memory)[:norm_elements])
        self.memory = new_deque


        if len(self.memory) < batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
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
    
    def train_step(self, batch_size: int):
        """Train the network using a batch of experiences."""

        if len(self.memory) < batch_size:
            return

        # print(f"len(self.memory) {len(self.memory)}")

        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
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
