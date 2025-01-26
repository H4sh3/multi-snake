import time
from large_environment import SnakeEnvLarge
from renderer import Renderer

r = Renderer(grid_size=(50,50))

env = SnakeEnvLarge()
env.reset()


env.step(0)
env.step(0)
env.step(0)
env.step(0)

r.render(env.snake,[[0,5]])

time.sleep(3)