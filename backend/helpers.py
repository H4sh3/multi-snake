from player import Direction

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

def get_inputs(player, grid, food):
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
    cols, rows = len(grid), len(grid[0])  # Dimensions of the grid

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

    return mapped