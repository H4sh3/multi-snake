import torch
import numpy as np
from collections import deque
import random
from dqncnn import SnakeCNN
from environment import GameEnvironment
from player import Direction
from helpers import get_inputs_cnn

import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        
        # Convert to torch tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

def train_snake_ai(
    model,
    env,
    episodes=100000,
    batch_size=64,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    min_replay_size=1000,
    target_update=10
):
    # Create target network for stable training
    # target_model = SnakeCNN()
    # target_model.load_state_dict(model.state_dict())
    target_model = model
    
    optimizer, crit, shed = create_training_setup(model)
    replay_buffer = ReplayBuffer()
    epsilon = epsilon_start
    episode_rewards = []

    env.add_player("0")
    
    for episode in range(episodes):
        env.reset()
        state = get_inputs_cnn(env.players["0"], env.visual_grid(), env.food)
        episode_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_indx = random.randint(0, 3)  # Assuming 4 possible actions
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = model(state_tensor)
                    action_indx = torch.argmax(q_values).item()
            
            if action_indx == 0:
                action = Direction.UP
            elif action_indx == 1:
                action = Direction.DOWN
            elif action_indx == 2:
                action = Direction.LEFT
            elif action_indx == 3:
                action = Direction.RIGHT
            
            # Take action and observe next state
            state = get_inputs_cnn(env.players["0"], env.visual_grid(), env.food)
            done, _, reward = env.step({"0":action})
            next_state = get_inputs_cnn(env.players["0"], env.visual_grid(), env.food)
            episode_reward += reward
            
            # Store transition in replay buffer
            replay_buffer.push(state, action_indx, reward, next_state, done)
            state = next_state
            
            # Start training when we have enough samples
            if len(replay_buffer) > min_replay_size:
                # Sample from replay buffer
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Compute Q values
                current_q_values = model(states).gather(1, actions.unsqueeze(1))
                
                # Compute target Q values
                with torch.no_grad():
                    max_next_q_values = target_model(next_states).max(1)[0]
                    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
                
                # Compute loss and update model
                loss = torch.nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Update target network periodically
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Store episode reward
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
    
    return episode_rewards

def evaluate_model(model, env, num_episodes=10):
    total_rewards = []
    
    for episode in range(num_episodes):
        env.reset()
        state = get_inputs_cnn(env.players["0"], env.visual_grid(), env.food)
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                action_indx = torch.argmax(q_values).item()
            
            if action_indx == 0:
                action = Direction.UP
            elif action_indx == 1:
                action = Direction.DOWN
            elif action_indx == 2:
                action = Direction.LEFT
            elif action_indx == 3:
                action = Direction.RIGHT

            done, _, reward = env.step({"0":action})
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    print(f"Average evaluation reward: {avg_reward:.2f}")
    return avg_reward


class SnakeNet(nn.Module):
    def __init__(self, input_channels=3, board_size=15, n_actions=4):
        super(SnakeNet, self).__init__()
        
        # Input shape: (batch_size, 3, 15)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        conv_out_size = board_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * conv_out_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer with action probabilities
        x = F.softmax(self.fc3(x), dim=1)
        
        return x

def create_training_setup(model):
    config = {
        'learning_rate': 0.0005,
        'weight_decay': 1e-5,
        'scheduler_step_size': 50,
        'scheduler_gamma': 0.9,
        'optimizer_type': 'adam'
    }
    """
    Creates the training setup for the Snake neural network.
    
    Args:
        model: The neural network model
        config (dict, optional): Configuration dictionary with training parameters
            
    Returns:
        tuple: (optimizer, criterion, scheduler)
    """
    if config is None:
        config = {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'scheduler_step_size': 100,
            'scheduler_gamma': 0.95,
            'optimizer_type': 'adam'
        }
    
    # Select optimizer
    if config['optimizer_type'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer_type'].lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:  # default to SGD
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
    
    # Loss function for Q-learning
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['scheduler_step_size'],
        gamma=config['scheduler_gamma']
    )
    
    return optimizer, criterion, scheduler

def train_step(model, state, target_action, optimizer, criterion):
    optimizer.zero_grad()
    
    # Convert inputs to torch tensors if they aren't already
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state)
    if not isinstance(target_action, torch.Tensor):
        target_action = torch.LongTensor(target_action)
    
    # Ensure proper dimensions
    if len(state.shape) == 3:
        state = state.unsqueeze(0)  # Add batch dimension
    
    # Forward pass
    predictions = model(state)
    loss = criterion(predictions, target_action)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Example usage:

# Initialize environment and model
env = GameEnvironment(width=15, height=15)
model = SnakeCNN(width=15, height=15, num_actions=4)

# Train the model
rewards = train_snake_ai(model, env)

# Evaluate the trained model
evaluate_model(model, env)

# Save the trained model
torch.save(model.state_dict(), 'snake_model.pth')


