from collections import deque
from environment import GameEnvironment
from dqncnn import CNNDQNAgent  # Updated to use CNN-based DQNAgent
import numpy as np
from player import Direction
import matplotlib.pyplot as plt
from helpers import get_inputs_cnn
import torch
import config

# Gym class
class Gym:
    def __init__(self, env: GameEnvironment):
        self.env = env

def initialize_gym(env_width: int = 15, env_height: int = 15) -> Gym:
    """
    Initializes the game environment and the gym.
    
    Parameters:
    - env_width (int): Width of the game grid.
    - env_height (int): Height of the game grid.
    
    Returns:
    - Gym: The initialized gym instance.
    """
    # Create the game environment
    environment = GameEnvironment(width=env_width, height=env_height)
    
    # Initialize the gym with the environment
    gym = Gym(env=environment)
    
    return gym

# Initialize gym

ENV_W = 6
ENV_H = 6
gym = initialize_gym(env_width=ENV_W, env_height=ENV_H)

# CNN-based DQN agent
agent = CNNDQNAgent( width=ENV_W, height=ENV_H, num_actions=4,num_channels=3)  # Input shape matches grid size

# Run multiple episodes
losses = []
highscore = 0

max_episode = 500000
gym.env.add_player("0")

def process_frames(frames, gamma=0.9):
    return np.stack(frames, axis=0)

for episode in range(max_episode):
    if episode % 100 == 0:
        print(f"{episode} - {highscore}")
    done = False
    step = 0
    gym.env.reset()
    agent.frames = deque(maxlen=3)


    player = gym.env.players["0"]
    next_state = gym.env.get_inputs(player)
    agent.frames.append(next_state)
    gym.env.step({"0":3})
    step += 1

    next_state = gym.env.get_inputs(player)
    agent.frames.append(next_state)
    gym.env.step({"0":3})
    step += 1

    next_state = gym.env.get_inputs(player)
    agent.frames.append(next_state)
    gym.env.step({"0":3})
    step += 1

    while not done and step < 1000:
        # Stack the states from the buffer
        # Weighted combination of the states
        player = gym.env.players["0"]

        prev_state = gym.env.get_inputs(player)
        while len(agent.frames) < 3:
            agent.frames.append(prev_state)

        prev_frames = process_frames(agent.frames.copy())
        prev_state_tensor = torch.tensor(prev_frames, dtype=torch.float32).unsqueeze(0).to(config.device)
        
        # Agent decides action
        action_indx = agent.select_action(prev_state_tensor)

        # Step the environment
        game_active, winner, reward = gym.env.step({"0":action_indx})

        # get next state
        next_state = gym.env.get_inputs(player)
        state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(config.device)  # Add batch dimension

        # add to frames
        agent.frames.append(next_state)

        after_frames = process_frames(agent.frames.copy())
        
        done = not game_active

        # Store experience
        agent.store_short_memory(prev_frames, action_indx, reward, after_frames, done)

        if reward == 1 and False:
            print("prev_frames")
            print(prev_frames)
            print("after_frames")
            print(after_frames)

        step += 1
    
    reward_cnt = sum(x[2] for x in agent.short_memory)    

    if reward_cnt > highscore:
        highscore = reward_cnt
        print(f"Highscore: {highscore}")

        model_save_path = f'checkpoints/cnn_checkpoint_{episode}_{highscore}.pth'
        agent.save_model(model_save_path)

    # Process experience
    agent.process_short_memory()

    # Train the agent
    loss = agent.train_step(batch_size=64)
    if loss:
        losses.append(float(loss))

    # Update target network and decay epsilon
    agent.update_target_network()
    agent.decay_epsilon(episode, (max_episode % 50000) + 1)

# Plot loss
fig, ax = plt.subplots()
plt.plot([i for i in range(len(losses))], losses)
plt.show()
