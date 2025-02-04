import json
import time
import os

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from env_3d_v2 import SnakeEnvLarge

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

        #if self.num_timesteps % self.check_freq == 0:
        rewards = [info["episode"]["r"] for info in self.locals["infos"] if "episode" in info]
        if len(rewards) > 0:
            mean_reward = sum(rewards) / len(rewards)
            if mean_reward >= self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.save_path, f"best_model_{int(mean_reward)}_ts_{self.num_timesteps}"))
        return True

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256, n_additional_inputs: int = 4):
        super().__init__(observation_space, features_dim)
        
        # Get grid dimensions from observation space
        grid_shape = observation_space["grid"].shape  # (T, H, W, C)
        self.frame_stack = grid_shape[0]
        in_channels = grid_shape[-1]  # 3 color channels
        
        # Adjusted 3D CNN architecture
        self.cnn3d = nn.Sequential(
            # Input shape: (batch, C, T, H, W)
            nn.Conv3d(in_channels, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # Output: (16, 4, 5, 5)
            
            nn.Conv3d(16, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),  # Output: (32, 4, 5, 5)
            
            nn.Conv3d(32, 64, kernel_size=(4, 3, 3), padding=(0, 1, 1)),  # Temporal convolution
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute output shape
        with th.no_grad():
            sample_grid = th.as_tensor(observation_space["grid"].sample()[None]).float()
            sample_grid = sample_grid.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
            n_flatten = self.cnn3d(sample_grid).shape[1]

        # Adjusted linear layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + n_additional_inputs, features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> th.Tensor:
        # Process grid observation
        grid = observations["grid"].float().permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        cnn_features = self.cnn3d(grid)
        
        # Process additional inputs
        additional_features = observations["additional_inputs"].float()
        
        # Concatenate and process
        combined = th.cat([cnn_features, additional_features], dim=1)
        return self.linear(combined)


class CustomPPO(PPO):
    def forward(self, obs, deterministic=False):
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


def make_env(seed):
    def _f():
        env = SnakeEnvLarge()
        return env
    return _f

def store_settings(settings, save_path):

    os.makedirs(save_path, exist_ok=True)

    with open(f"{save_path}/config.json", "w+") as f:
        f.write(json.dumps(settings,indent=2))

def optimize_ppo(trial=None):
    env = make_vec_env(SnakeEnvLarge, n_envs=64)#, vec_env_cls=SubprocVecEnv)
    Algo = CustomPPO

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            features_dim=256,
            n_additional_inputs=4
        )
    )

    # learned max for 8x8
    if trial:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [512, 1024, 1536])
        gamma = trial.suggest_float("gamma", 0.9, 0.999)
        vf_coef = trial.suggest_float("gamma", 0.9, 0.999)
        ent_coef = trial.suggest_float("epsilon_decay", 0.4, 0.6)
        n_steps = trial.suggest_int("replay_buffer_size", 1000, 5000)
        n_epochs = trial.suggest_int("target_update_frequency", 10, 20)
        clip_range = trial.suggest_float("trial.suggest_float", 0.10, 0.30)
    
        model = PPO(
            "MultiInputPolicy", 
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./tensorboard_ppo_snake/",
            learning_rate=learning_rate,
            gamma=gamma,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            n_steps=n_steps,
            clip_range=clip_range,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )

        foldername = f"{learning_rate}_{batch_size}_{gamma}_{vf_coef}_{ent_coef}_{n_steps}_{n_epochs}_{clip_range}"
    else:
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
            n_steps=1024*3,
            clip_range=0.25,
            n_epochs = 20,
            batch_size= 1024,
        )

        config = {
            "verbose":1,
            "learning_rate":1e-3,
            "gamma":0.99,
            "ent_coef":0.01,
            "vf_coef":0.6,
            "n_steps":1024,
            "clip_range":0.2,
            "n_epochs":15,
            "batch_size":512,
        }

        config = {
            "verbose": 1,
            "learning_rate": 1e-4,
            "gamma": 0.95,
            "ent_coef": 0.01,
            "vf_coef": 0.6,
            "n_steps": 1024*2,
            "clip_range": 0.2,
            "n_epochs": 15,
            "batch_size": 1024,
        }

        model = Algo(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./tensorboard_ppo_snake/",
            verbose=1,
            learning_rate=config["learning_rate"], 
            gamma=config["gamma"], 
            ent_coef=config["ent_coef"], 
            vf_coef=config["vf_coef"], 
            n_steps=config["n_steps"], 
            clip_range=config["clip_range"], 
            n_epochs=config["n_epochs"], 
            batch_size=config["batch_size"], 
        )
        foldername = int(time.time()*1000000)
    
    print(f"training {foldername}")

    save_path = f"./checkpoints/optuna_{foldername}"

    store_settings(config,save_path)

    # continue training
    # model_name = f'{save_path}/best_model_51_ts_18928000.zip'
    # model = Algo.load(model_name)
    # model.set_env(env)

    model_name = f'{save_path}/net'

    callback = SaveOnBestTrainingRewardCallback(save_path=save_path,check_freq=10000)

    model.learn(
        total_timesteps=50_000_000,
        callback = callback
        )

    model.save(model_name)

    return callback.best_mean_reward


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
        env.render()
        time.sleep(1/20)

        if done:
            print(f"Score: {score} Highscore: {highscore}")

            if score > highscore:
                highscore = score
            score = 0
            obs = env.reset()[0]


if __name__ == "__main__":
    # study = optuna.create_study(direction="maximize")  # Maximize average reward
    # study.optimize(optimize_ppo, n_trials=10)
    optimize_ppo()