from stable_baselines3 import PPO
import torch as th
import torch.nn as nn
from gymnasium import spaces 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from neuralnetwork.environment import SnakeEnv

from renderer import Renderer

# model_name = './checkpoints/15_01_2025/best_model_61.zip' # 8x8
model_name = './checkpoints/16_01_2025_10by10/best_model_90_ts_82920000.zip'
model_name = './checkpoints/16_01_2025_10by10/best_model_94_ts_104072000.zip'

model = PPO.load(model_name)
env = SnakeEnv()
renderer = Renderer()

obs = env.reset()[0]
highscore = 0
score = 0
step = 0
games_played = 0
total_score = 0

while True:
    action, _ = model.predict(obs)

    obs, reward, done, _, info = env.step(int(action))
    if reward == 1:
        score += reward

    # renderer manages fps with pygame
    renderer.render(env.snake, env.food)

    # Calculate metrics
    mean_score = total_score / games_played if games_played > 0 else 0
    golden_ratio = highscore / games_played if games_played > 0 else 0

    # Enhanced metrics print
    print(f"Highscore: {highscore}")
    print(f"Current Score: {score}")
    print(f"Steps: {step}")
    print(f"Mean Score: {mean_score:.2f}")
    print(f"Golden Ratio (Highscore/Games Played): {golden_ratio:.2f}")
    print(f"Action Taken: {action}")
    print(f"Game State Info: {info}")  # Assuming `info` contains additional environment details

    step += 1
    if done:
        print(f"Game Over!")
        print(f"Final Score: {score}")
        print(f"Total Steps: {step}")
        games_played += 1
        total_score += score

        if score > highscore:
            highscore = score
            print("New Highscore!")

        score = 0
        obs = env.reset()[0]
        step = 0
