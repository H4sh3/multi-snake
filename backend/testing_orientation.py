from environment import GameEnvironment
from player import Direction
from helpers import get_inputs
from contextlib import contextmanager
import signal

# Timeout context manager for identifying hanging operations
@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

def inputs_tests():
    env = GameEnvironment()
    env.add_player("0")
    env.step({"0":Direction.RIGHT})
    print(env.players["0"].x)
    print(env.players["0"].y)

    # UP
    env.players["0"].direction = Direction.UP
    env.spawn_food(6,4)
    inputs = get_inputs(env.players["0"], env.visual_grid(), env.food)
    assert inputs[3] == 1
    assert inputs[9] == 1

    # RIGHT
    env.players["0"].direction = Direction.RIGHT
    env.spawn_food(7,5)
    inputs = get_inputs(env.players["0"], env.visual_grid(), env.food)
    assert inputs[6] == 1
    assert inputs[9] == 1

    # DOWN
    env.players["0"].direction = Direction.DOWN
    env.spawn_food(6,6)
    inputs = get_inputs(env.players["0"], env.visual_grid(), env.food)
    assert inputs[4] == 1
    assert inputs[9] == 1


    # LEFT
    env.players["0"].direction = Direction.LEFT
    env.spawn_food(5,5)
    inputs = get_inputs(env.players["0"], env.visual_grid(), env.food)
    print(inputs)
    assert inputs[9] == 1
    #assert inputs[16] == 1


def env_test():
    env = GameEnvironment()
    env.add_player("0")

    env.step({"0":Direction.RIGHT})
    env.step({"0":Direction.RIGHT})
    env.step({"0":Direction.RIGHT})
    env.step({"0":Direction.RIGHT})
    env.step({"0":Direction.RIGHT})
    env.step({"0":Direction.RIGHT})
    env.step({"0":Direction.DOWN})
    env.step({"0":Direction.DOWN})

    print(env.grid)
    env.render_game_state()




if __name__ == "__main__":
    env_test()