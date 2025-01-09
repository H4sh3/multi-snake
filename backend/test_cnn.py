from environment import GameEnvironment
from player import Direction
from helpers import get_inputs_cnn
import numpy as np
import torch
from dqncnn import SnakeCNN
import time
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

def initialize_game():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = GameEnvironment(6,6)
    env.add_player("0")
    
    # Load model
    model_path = "checkpoints/cnn_checkpoint_66365_8.pth"

    model = SnakeCNN(6,6,4).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    model.eval()
    return env, model, device

def main():
    env, model, device = initialize_game()
    frame_count = 0
    game_active = True
    
    try:
        reward_sum = 0
        while game_active:
            frame_start_time = time.time()
            
            try:
                with timeout(1):  # Set 1-second timeout for game operations
                    # Get game state
                    player = env.players["0"]
                    inputs = get_inputs_cnn(player, env.visual_grid(), env.food)
                    state = np.array(inputs, dtype=np.float32)
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Model inference
                    with torch.no_grad():
                        action_idx = model.forward(state_tensor).argmax().item()
                    
                    # Convert action index to direction
                    action = {
                        0: Direction.UP,
                        1: Direction.DOWN,
                        2: Direction.LEFT,
                        3: Direction.RIGHT
                    }[action_idx]

                    print(f"action {action_idx}")
                    
                    # Update game state
                    game_active, winner, reward = env.step({"0": action})
                    reward_sum += reward

                    if not game_active:
                        print(inputs)
                    env.render_game_state()
                    print(f"Score: {reward_sum}")
                    
                    # Calculate and enforce frame timing
                    frame_time = time.time() - frame_start_time
                    sleep_time = max(0.1 - frame_time, 0)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                    frame_count += 1
                    print(frame_count)
                    if frame_count % 100 == 0:
                        print(f"Processed {frame_count} frames")
            
            except TimeoutError:
                print("Game operation timed out - resetting frame")
                continue
            except Exception as e:
                print(f"Error during game loop: {e}")
                raise

            if not game_active:
                reward_sum = 0
                env.reset()
                game_active = True
                
    except KeyboardInterrupt:
        print("\nGame terminated by user")
    finally:
        print(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    main()