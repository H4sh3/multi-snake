from stable_baselines3 import A2C, PPO, DQN
import time
from stable_baselines3.common.env_util import make_vec_env
from gym_env import GameEnvironment
import torch as th
import torch.nn as nn
from gymnasium import spaces, Env
import numpy as np

from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

from stable_baselines3.common.env_checker import check_env
#from gym.envs.registration import register
#register(
#    id='GameEnvironment-v0',  # Unique identifier for your environment
#    entry_point='my_gym_envs.game_env:GameEnvironment',  # Path to your environment class
#)
#
## Parallel environments
#vec_env = make_vec_env("GameEnvironment-v0", n_envs=4)


class VecExtractDictObs(VecEnvWrapper):
    """
    A vectorized wrapper for filtering a specific key from dictionary observations.
    Similar to Gym's FilterObservation wrapper:
        https://github.com/openai/gym/blob/master/gym/wrappers/filter_observation.py

    :param venv: The vectorized environment
    :param key: The key of the dictionary observation
    """

    def __init__(self, venv: VecEnv, key: str):
        self.key = key
        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs[self.key]

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs[self.key], reward, done, info

class CustomCNN(VecEnv):
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

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=64),
)


env = GameEnvironment(10,10)
stacked_env = VecExtractDictObs(env,'x0')


model = PPO(
        "CnnPolicy",
        stacked_env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1
    )

#policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict([3,6,6]))
#model = A2C("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

model.learn(total_timesteps=500_000)
model_name = 'SNAK'
model.save(model_name)

del model # remove to demonstrate saving and loading

model = PPO.load(model_name)

obs = env.reset()[0]
highscore = 0
score = 0
while True:
    action, _states = model.predict(obs)
    obs, reward, done, _, info = env.step(int(action))
    if reward == 1:
        score += reward
    print(f"obs.shape1 {obs.shape}")
    env.render("human")

    if done:
        print(f"Score: {score} Highscore: {highscore}")
        time.sleep(1)

        if score > highscore:
            highscore = score
        score = 0
        obs = env.reset()[0]
