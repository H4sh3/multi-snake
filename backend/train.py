import random
from collections import deque
from environment import GameEnvironment
from typing import Tuple, Dict
from dqn import DQNAgent
import numpy as np
from player import Direction
import matplotlib.pyplot as plt
from helpers import get_inputs

# Gym class
class Gym:
    def __init__(self, env: GameEnvironment, memory_size: int = 1000):
        self.env = env

def initialize_gym(env_width: int = 15, env_height: int = 15, memory_size: int = 1000) -> Gym:
    """
    Initializes the game environment and the gym.
    
    Parameters:
    - env_width (int): Width of the game grid.
    - env_height (int): Height of the game grid.
    - memory_size (int): Maximum capacity of the memory.
    
    Returns:
    - Gym: The initialized gym instance.
    """
    # Create the game environment
    environment = GameEnvironment(width=env_width, height=env_height)
    
    # Initialize the gym with the environment and memory size
    gym = Gym(env=environment, memory_size=memory_size)
    
    return gym

# Initialize gym
gym = initialize_gym(env_width=15, env_height=15, memory_size=1000)

# DQN agent
input_shape = 24  # Grid dimensions
num_actions = 4  # ["up", "down", "left", "right", "none"]

agent = DQNAgent(input_shape, num_actions)

# Run multiple episodes

losses = []

highscore = 0

max_episode = 50000
gym.env.add_player("0")

for episode in range(max_episode):
    if episode % 100 == 0:
        print(f"{episode} - {highscore}")
    gym.env.reset()

    done = False
    step = 0

    while not done and step < 1000:
        # print(f"Step: {step}")
        
        # Agent decides action
        player = gym.env.players["0"]
        inputs = get_inputs(player, gym.env.visual_grid(), gym.env.food)
        state = np.array(inputs, dtype=np.float32)

        action_indx = agent.select_action(state)
        if action_indx == 0:
            action = Direction.UP
        elif action_indx == 1:
            action = Direction.DOWN
        elif action_indx == 2:
            action = Direction.LEFT
        elif action_indx == 3:
            action = Direction.RIGHT
        elif action_indx == 4:
            action = None

        # Step the environment
        prev_state = get_inputs(gym.env.players["0"],gym.env.visual_grid(),gym.env.food)
        game_active, winner, reward = gym.env.step({"0": action})
        next_state = get_inputs(gym.env.players["0"],gym.env.visual_grid(),gym.env.food)
        # Calculate rewards
        reward = 1 if reward else 0
        done = not game_active

        # Store experience
        agent.store_short_memory(prev_state, action_indx, reward, next_state, done)
        step += 1
    
    reward_cnt = sum(x[2] for x in agent.short_memory)    

    if reward_cnt > highscore:
        highscore = reward_cnt
        print(f"Highscore: {highscore}")

        model_save_path = f'checkpoints/checkpoint_{episode}_{highscore}.pth'
        agent.save_model(model_save_path)

    # agent is done - process experience
    agent.process_short_memory()


    # Train the agent
    loss = agent.train_step(batch_size=32)
    if loss:
        losses.append(float(loss))

    # print(f"Episode {episode}: {len(losses)}")
    # Update target network and decay epsilon
    agent.update_target_network()
    agent.decay_epsilon(episode, max_episode)

    if False:#if episode % 1000 == 0:
        model_save_path = f'checkpoints/checkpoint_{episode}.pth'
        agent.save_model(model_save_path)

fig, ax = plt.subplots()
plt.plot([i for i in range(0,len(losses))],losses)
plt.show()


