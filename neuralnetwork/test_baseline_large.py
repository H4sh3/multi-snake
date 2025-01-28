from stable_baselines3 import PPO
import torch as th
import torch.nn as nn
from gymnasium import spaces 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from large_env_deep import SnakeEnvLarge
from renderer import Renderer
import time

model_name = './checkpoints/optuna_1738094466442004/best_model_195_ts_2777152'

model = PPO.load(model_name)
env = SnakeEnvLarge()
renderer = Renderer(grid_size=(20,20), cell_size=50)

obs = env.reset()[0]
highscore = 0
score = 0
step = 0
games_played = 0
total_score = 0

# Initialize a list to store the last 15 moves
replay_buffer = []

n = 0


while True:
    action, _ = model.predict(obs)

    # Store the current state, action, snake, and food in the replay buffer
    replay_buffer.append({
        'obs': obs.copy(),
        'action': action,
        'snake': env.snake.copy(),  # Store the snake's state
        'food': env.food.copy()     # Store the food's state
    })

    # If the buffer exceeds 15 moves, remove the oldest move
    if len(replay_buffer) > 15:
        replay_buffer.pop(0)

    print(f"action {action}")
    obs, reward, done, _, info = env.step(int(action))
    if reward == 1:
        score += reward

    # renderer manages fps with pygame
    renderer.render(env.snake, env.food, obs)

    # Calculate metrics
    mean_score = total_score / games_played if games_played > 0 else 0
    golden_ratio = highscore / games_played if games_played > 0 else 0

    # Enhanced metrics print
    #print(f"Highscore: {highscore}")
    #print(f"Current Score: {score}")
    #print(f"Steps: {step}")
    #print(f"Mean Score: {mean_score:.2f}")
    #print(f"Golden Ratio (Highscore/Games Played): {golden_ratio:.2f}")

    #print(f"Game State Info: {info}")  # Assuming `info` contains additional environment details

    step += 1
    if done:
        n+=1
        print(f"Game Over!")

        renderer.render(env.snake, env.food, obs)
        renderer.save(n,score)
        # time.sleep(6)
        print(f"Final Score: {score}")
        print(f"Total Steps: {step}")
        games_played += 1
        total_score += score

        if score > highscore:
            highscore = score
            print("New Highscore!")

        # Replay the last 15 moves
        print("Replaying last 15 moves...")
        for move in replay_buffer:
            break
            # Restore the snake and food positions for rendering
            env.snake = move['snake'].copy()
            env.food = move['food'].copy()
            renderer.render(env.snake, env.food)
            print(f"Replay Action: {move['action']}")
        

        # Reset the replay buffer
        replay_buffer = []

        score = 0
        obs = env.reset()[0]
        step = 0