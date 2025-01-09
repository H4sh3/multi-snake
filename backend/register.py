from gym.envs.registration import register

register(
    id='GameEnvironment-v0',  # Unique identifier for your environment
    entry_point='./gym_env:GameEnvironment',  # Path to your environment class
)