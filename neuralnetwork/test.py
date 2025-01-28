import time
from large_env_deep import SnakeEnvLarge
from renderer import Renderer

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