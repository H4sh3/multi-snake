from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import torch.nn as nn

from large_environment import SnakeEnvLarge
from gymnasium import spaces 

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

import time
import os

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Custom callback for saving a model when the training reward is improved.
    """
    def __init__(self, save_path: str, check_freq: int = 1000):
        super(SaveOnBestTrainingRewardCallback, self).__init__()
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_mean_reward = -float("inf")
        self.n_calls = 0

    def _init_callback(self) -> None:
        # Ensure the save directory exists
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check the training reward every `check_freq` steps
        if self.n_calls % self.check_freq == 0:
            # Compute the mean reward
            if True:#len(self.locals["infos"]) > 0 and "episode" in self.locals["infos"][0]:
                rewards = [info["episode"]["r"] for info in self.locals["infos"] if "episode" in info]
                if len(rewards) > 0:
                    mean_reward = sum(rewards) / len(rewards)
                    # print(f"Step {self.n_calls}: Mean reward: {mean_reward:.2f}")
                    
                    # Save the model if the reward is improved
                    if mean_reward >= self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        print(f"New best reward! Saving model to {self.save_path} ts {self.num_timesteps}")
                        self.model.save(os.path.join(self.save_path, f"best_model_{int(mean_reward)}_ts_{self.num_timesteps}"))

        return True

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of units for the last layer.
    :param n_additional_inputs: (int) Number of additional inputs to be merged.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256, n_additional_inputs: int = 4):
        super().__init__(observation_space, features_dim)
        extractors = {}

        # We assume CxHxW images (channels first)
        n_input_channels = observation_space["grid"].shape[0]
        # Define CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space["grid"].sample()[None]).float()
            ).shape[1]

        # Define linear layers
        # Combine flattened CNN output with additional inputs
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + n_additional_inputs, features_dim),
            nn.ReLU()
        )


    def forward(self, observations) -> th.Tensor:
        """
        Forward pass with additional inputs merged.
        :param observations: (th.Tensor) Image-based observations.
        :param additional_inputs: (th.Tensor) Additional inputs to be merged.
        :return: (th.Tensor) Output features.
        """
        print(observations)
        cnn_features = self.cnn(observations[0])
        # Concatenate CNN output with additional inputs
        combined_features = th.cat([cnn_features, observations[1]], dim=1)
        return self.linear(combined_features)


class CustomPPO(PPO):
    """
    Custom Policy with a CNN and additional inputs.
    """
    #def __init__(self, policy, env, policy_kwargs,learning_rate,gamma,ent_coef,vf_coef,verbose,tensorboard_log,n_steps,clip_range,n_epochs,batch_size, **kwargs):
    #    super(PPO, self).__init__(
    #        policy=policy,
    #        env=env,
    #        policy_kwargs = policy_kwargs,
    #        learning_rate = learning_rate,
    #        gamma = gamma,
    #        ent_coef = ent_coef,
    #        vf_coef = vf_coef,
    #        verbose = verbose,
    #        tensorboard_log = tensorboard_log,
    #        n_steps = n_steps,
    #        #clip_range = clip_range,
    #        #n_epochs = n_epochs,
    #        #batch_size = batch_size,
    #        gae_lambda = 0.95,
    #        max_grad_norm = 0.5,
    #        use_sde = False,
    #        sde_sample_freq = -1,
    #        **kwargs
    #        )

    def forward(self, obs, deterministic=False):
        print(obs)
        cnn_features = self.cnn(obs[0])
        # Concatenate CNN output with additional inputs
        combined_features = th.cat([cnn_features, obs[1]], dim=1)

        # Pass through the policy and value networks
        distribution = self._get_action_dist_from_latent(combined_features)
        values = self.value_net(combined_features)

        return distribution, values

    def _predict(self, obs, deterministic=False):
        distribution, _ = self.forward(obs, deterministic=deterministic)
        return distribution.get_actions(deterministic=deterministic)

def optimize_ppo():
    env = make_vec_env(SnakeEnvLarge, n_envs=32)

    Algo = CustomPPO

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=4),
    )

    # learned max for 8x8
    model = Algo(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log="./tensorboard_ppo_snake/",
        n_steps=3072,
        clip_range=0.25,
        n_epochs = 20,
        batch_size= 1024,
    )

    save_path = "./checkpoints/large_18_01_2025_10by10"

    # continue training
    #model_name = f'{save_path}/best_model_51_ts_18928000.zip'
    #model = Algo.load(model_name)
    #model.set_env(env)

    model_name = f'{save_path}/net'

    model.learn(
        total_timesteps=250_000_000,
        callback = SaveOnBestTrainingRewardCallback(save_path=save_path,check_freq=250)
        )
    model.save(model_name)

    del model # remove to demonstrate saving and loading

    model = Algo.load(model_name)

    env = SnakeEnvLarge()

    obs = env.reset()[0]
    highscore = 0
    score = 0

    while True:
        action, _states = model.predict(obs)
        obs, reward, done, _, info = env.step(int(action))
        if reward == 1:
            score += reward
        # print(f"obs.shape1 {obs.shape}")
        env.render()
        time.sleep(1/20)

        if done:
            print(f"Score: {score} Highscore: {highscore}")

            if score > highscore:
                highscore = score
            score = 0
            obs = env.reset()[0]


if __name__ == "__main__":
    optimize_ppo()