from backend.player import Direction
import numpy as np

def get_inputs_rotated(player, grid, food):
    """
    Returns the 8 neighbors of a given cell (x, y) in a 2D grid, wrapping around the edges.

    :param player: Player object with x and y coordinates.
    :param grid: 2D list representing the grid (10x10).
    :param food: Tuple representing the food's position (x, y).
    :return: List of neighbor values and other relevant information.
    """

    neighbors = []
    directions = [
        (-1, -1), (-1, 0), (-1, 1),  # Top-left, Top, Top-right
        (0, -1),          (0, 1),   # Left,       Right
        (1, -1), (1, 0), (1, 1)     # Bottom-left, Bottom, Bottom-right
    ]
    directions = [
        (0, -1), (1, -1), (1, 0),  # Top-left, Top, Top-right
        (1, 1),          (0, 1),   # Left,       Right
        (-1, 1), (-1, 0), (-1, 1)     # Bottom-left, Bottom, Bottom-right
    ]
    cols, rows = len(grid), len(grid[0])  # Dimensions of the grid

    dir_translated = []
    if player.direction == Direction.UP:
        dir_translated = directions
    elif player.direction == Direction.RIGHT:
        dir_translated = directions[2:] + directions[:2]
    elif player.direction == Direction.DOWN:
        dir_translated = directions[4:] + directions[:4]
    elif player.direction == Direction.LEFT:
        dir_translated = directions[6:] + directions[:6]
    directions = dir_translated

    for dx, dy in directions:
        # Wrap around using modulo operator
        nx = (player.x + dx) % rows
        ny = (player.y + dy) % cols
        neighbors.append(grid[nx][ny])

    mapped = []


    # Food
    for v in neighbors:
        if v == 'F':
            mapped.append(1)
        else:
            mapped.append(0)

    # Body
    for v in neighbors:
        if v == 'x':
            mapped.append(1)
        else:
            mapped.append(0)


    # Food direction
    if food is not None:
        food_direction = [
            1 if player.y > food[1] else 0, # TOP
            1 if player.x < food[0] else 0, # RIGHT
            1 if player.y < food[1] else 0, # DOWN
            1 if player.x > food[0] else 0, # LEFT
        ]
        #mapped += food_direction
    else:
        food_direction = [0,0,0,0]

    food_dir_translated = []
    if player.direction == Direction.UP:
        food_dir_translated = food_direction
    elif player.direction == Direction.RIGHT:
        food_dir_translated = food_direction[1:] + food_direction[:1]
    elif player.direction == Direction.DOWN:
        food_dir_translated = food_direction[2:] + food_direction[:2]
    elif player.direction == Direction.LEFT:
        food_dir_translated = food_direction[3:] + food_direction[:3]

    mapped += food_dir_translated


    return mapped

DEBUG = False
def get_inputs(player, grid, food):
    """
    Returns the 8 neighbors of a given cell (x, y) in a 2D grid, wrapping around the edges.

    3*3 = 9 - 1 = 8
    4*4 = 16 -1 = 15
    5*5 = 25 - 1 = 24
    6*6 = 36 - 1 = 35

    :param player: Player object with x and y coordinates.
    :param grid: 2D list representing the grid (10x10).
    :param food: Tuple representing the food's position (x, y).
    :return: List of neighbor values and other relevant information.
    """
    neighbors = []

    directions = []
    scan_size = 1
    for oX in range(-scan_size,scan_size+1):
        for oY in range(-scan_size,scan_size+1):
            if oX == 0 and oY == 0:
                continue
            
            directions.append((oX,oY))

    cols, rows = len(grid), len(grid[0])  # Dimensions of the grid
    for dx, dy in directions:
        # Wrap around using modulo operator
        nx = player.x + dx
        ny = player.y + dy

        # check out of bounds
        if nx > cols-1 or nx < 0 or ny > rows-1 or ny < 0:
            neighbors.append("w") # not good like the body
        else:
            neighbors.append(grid[nx][ny])


    mapped = []

    # Food
    for v in neighbors:
        if v == 'F':
            mapped.append(1)
        else:
            mapped.append(0)

    # Body
    sum_body = 0
    for v in neighbors:
        if v == 'x' or v == 'w':
            mapped.append(1)
        else:
            mapped.append(0)

    if DEBUG:
        print(f"sum_body {sum_body}")


    # Food direction
    if food is not None:
        food_direction = [
            1 if player.y > food[1] else 0, # TOP
            1 if player.x < food[0] else 0, # RIGHT
            1 if player.y < food[1] else 0, # DOWN
            1 if player.x > food[0] else 0, # LEFT
        ]
        #mapped += food_direction
    else:
        food_direction = [0,0,0,0]


    mapped += food_direction

    # direction
    if player.direction == Direction.UP:
        mapped += [1,0,0,0]
    elif player.direction == Direction.RIGHT:
        mapped += [0,1,0,0]
    elif player.direction == Direction.DOWN:
        mapped += [0,0,1,0]
    elif player.direction == Direction.LEFT:
        mapped += [0,0,0,1]

    # print(f"len mapped {len(mapped)}")
    return mapped


def get_inputs_cnn(player, grid, food):
    """
    Prepares the input grid for a CNN, encoding game state information into separate channels.

    :param player: Player object with x, y coordinates and direction.
    :param grid: 2D list representing the game grid (e.g., 10x10).
    :param food: Tuple representing the food's position (x, y).
    :return: 3D NumPy array of shape (channels, height, width).
    """
    # Dimensions of the grid
    height, width = len(grid), len(grid[0])
    
    # Initialize channels
    player_channel = np.zeros((height, width), dtype=np.float32)  # Player's position
    body_channel = np.zeros((height, width), dtype=np.float32)    # Snake's body
    food_channel = np.zeros((height, width), dtype=np.float32)    # Food's position
    
    # Encode player position
    player_channel[player.x, player.y] = 1.0

    # Encode snake's body
    for x in range(height):
        for y in range(width):
            if grid[x][y] == 'x':  # Assuming 'x' represents the snake's body
                body_channel[x, y] = 1.0
    
    # Encode food position
    if food is not None:
        food_channel[food[0], food[1]] = 1.0

    # Stack the channels into a 3D array
    cnn_input = np.stack([player_channel, body_channel, food_channel], axis=0)
    
    return cnn_input


def render(snakeGameEnv, mode='human'):
    """
    Renders the game state.
    Mode options:
    - 'human': Prints to console with colors (if supported)
    - 'rgb_array': Returns numpy array (not implemented)
    """
    try:
        from colorama import init, Fore, Back, Style
        init()  # Initialize colorama
        use_color = True
    except ImportError:
        use_color = False

    # Create empty board
    board = [[' ' for _ in range(snakeGameEnv.width)] for _ in range(snakeGameEnv.height)]
    
    # Place snake body
    for i, (x, y) in enumerate(snakeGameEnv.snake):
        if i == 0:  # Head
            symbol = '█'  # Solid block for head
        else:  # Body
            symbol = '▓'  # Different block for body
        board[x][y] = symbol
    
    # Place food
    board[snakeGameEnv.food[0]][snakeGameEnv.food[1]] = '●'  # Circle for food
    
    # Print the board
    print("\n" + "═" * (snakeGameEnv.width * 2 + 2))  # Top border
    
    for row in board:
        print("║", end="")  # Left border
        for cell in row:
            if use_color:
                if cell == '█':  # Snake head
                    print(Fore.GREEN + cell + Style.RESET_ALL + " ", end="")
                elif cell == '▓':  # Snake body
                    print(Fore.CYAN + cell + Style.RESET_ALL + " ", end="")
                elif cell == '●':  # Food
                    print(Fore.RED + cell + Style.RESET_ALL + " ", end="")
                else:  # Empty space
                    print(cell + " ", end="")
            else:
                print(cell + " ", end="")
        print("║")  # Right border
    
    print("═" * (snakeGameEnv.width * 2 + 2))  # Bottom border
    
    # Print game info
    print(f"Score: {snakeGameEnv.score}")
    print(f"Steps without food: {snakeGameEnv.steps_without_food}")

    if use_color:
        print(Style.RESET_ALL)  # Reset all colors