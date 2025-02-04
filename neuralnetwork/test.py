import time
from large_env_deep import SnakeEnvLarge
from renderer import Renderer
from double_conv_env import SnakeEnvLarge

env = SnakeEnvLarge()
env.reset()
obs = env._get_observation()
print(obs["grid"])
print(obs["grid"].shape)
print(obs["low_res_grid"].shape)
exit()


r = Renderer(grid_size=(50,50))

env = SnakeEnvLarge()
env.reset()


env.step(0)
env.step(0)
env.step(0)
env.step(0)

obs = env._get_observation()
print(obs)


r.render(env.snake,[[0,5]], obs)

time.sleep(5)