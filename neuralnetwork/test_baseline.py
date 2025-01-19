from stable_baselines3 import PPO
import torch as th
import torch.nn as nn
from gymnasium import spaces 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from neuralnetwork.environment import SnakeEnv

from renderer import Renderer

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256,n_stack=3,num_envs=3):
        print(observation_space)
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

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
