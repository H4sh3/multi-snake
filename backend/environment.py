
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Tuple, Union
import random
from player import Player, Direction
from helpers import get_inputs

def get_neighbors(grid, x, y):
    """
    Returns the 8 neighbors of a given cell (x, y) in a 2D grid.
    
    :param grid: 2D list representing the grid (10x10).
    :param x: x-coordinate (row index) of the cell.
    :param y: y-coordinate (column index) of the cell.
    :return: List of neighbor values.
    """
    neighbors = []
    directions = [
        (-1, -1), (-1, 0), (-1, 1),  # Top-left, Top, Top-right
        (0, -1),          (0, 1),   # Left,       Right
        (1, -1), (1, 0), (1, 1)     # Bottom-left, Bottom, Bottom-right
    ]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 10 and 0 <= ny < 10:  # Check bounds
            neighbors.append(grid[nx][ny])
    
    return neighbors

class GameEnvironment:
    def __init__(self, width: int = 15, height: int = 15):
        self.width = width
        self.height = height
        self.players: Dict[str, Player] = {}
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.food: Tuple[int, int] = 0,0
        self.spawn_food()

        self.game_active = False

    def spawn_food(self,x_spawn:int = None,y_spawn:int = None) -> Tuple[int, int]:
        while True:
            x = x_spawn if x_spawn is not None else random.randint(0, self.width - 1)
            y = y_spawn if x_spawn is not None else random.randint(0, self.height - 1)
            # print(f"food spawn at {x}{y}")
            if self.grid[x][y] is None:
                self.grid[x][y] = "FOOD"
                self.food = x, y
                return

    def add_player(self, sid: str) -> Union[Player, None]:
        if len(self.players) >= 4:
            return None

        positions = [
            (5, 5, Direction.RIGHT, "#FF0000"),
            (self.width - 5, self.height - 5, Direction.LEFT, "#0000FF"),
            (5, self.height - 5, Direction.RIGHT, "#00FF00"),
            (self.width - 5, 5, Direction.LEFT, "#FFFF00")
        ]
        pos = positions[len(self.players)]
        
        player = Player(*pos)
        player.sid = sid
        player.trail = [pos[:2]] * 5
        self.players[sid] = player
        self.grid[player.y][player.x] = player.color
        return player

    def reset(self):
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        positions = [
            (5, 5, Direction.RIGHT, "#FF0000"),
            (self.width - 5, self.height - 5, Direction.LEFT, "#0000FF"),
            (5, self.height - 5, Direction.RIGHT, "#00FF00"),
            (self.width - 5, 5, Direction.LEFT, "#FFFF00")
        ]
        for i, player in enumerate(self.players.values()):
            player.alive = True
            player.ready = False
            x, y, direction, color = positions[i]
            player.x = x
            player.y = y
            player.direction = direction
            player.color = color
            player.trail = [(x, y)] * 5
            self.grid[y][x] = color
        self.spawn_food()
        self.game_active = False

    # returns 
    def step(self, actions: Dict[str, str]) -> Tuple[bool, Union[str, None], bool]:
        reward = 0

        for sid, action in actions.items():
            if action:
                self.turn_player(sid, action)

        alive_players = [p for p in self.players.values() if p.alive]
        if len(alive_players) == 0:
            return False, None, reward

        for player in alive_players:
            dx, dy = player.direction.value
            new_x = (player.x + dx) % self.width
            new_y = (player.y + dy) % self.height

            # tail crash logic
            if self.grid[new_y][new_x] is not None:
                #print("DEATHCAM")
                #self.render_game_state()
                reward = 0
                player.alive = False
                for x, y in player.trail:
                    self.grid[y][x] = None
                return False, None, reward

            # collected food
            if (new_x, new_y) == self.food:
                # print(get_inputs(self.players["0"], self.grid, self.food))
                reward = 1
                player.trail.append((new_x, new_y))
                self.spawn_food()
            else:
                tail_end = player.trail.pop(0)
                self.grid[tail_end[1]][tail_end[0]] = None

            player.x = new_x
            player.y = new_y
            player.trail.append((new_x, new_y))
            self.grid[new_y][new_x] = player.color

        return True, None, reward

    def turn_player(self, player_id: str, new_direction: Direction):
        if player_id not in self.players:
            return

        player = self.players[player_id]
        if not player.alive:
            return

        current = player.direction
        # should not turn on the spot
        if new_direction.value[0] + current.value[0] == 0 and new_direction.value[1] + current.value[1] == 0:
            return

        player.direction = new_direction

    def to_dict(self):
        return {
            "grid": self.grid,
            "players": {
                pid: {
                    "x": p.x,
                    "y": p.y,
                    "direction": p.direction.name,
                    "color": p.color,
                    "alive": p.alive,
                    "ready": p.ready,
                    "wins": p.wins,
                    "trail": p.trail
                }
                for pid, p in self.players.items()
            },
            "food": self.food,
            "active": self.game_active
        }

    def grid_as_input(self, agnt_color):
        # one grid for iteself.
        # one for enemies
        # one for food

        agent_pos_x = self.players[agnt_color].x
        agent_pos_y = self.players[agnt_color].y
        get_neighbors(self.grid,agent_pos_x,agent_pos_y)


        
    def render_game_state(self):
        """
        Renders the current game state to the terminal.

        :param game: The GameEnvironment instance to render.
        """
        # Initialize a 2D array representing the visual grid
        visual_grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Place food on the grid
        food_x, food_y = self.food
        visual_grid[food_y][food_x] = 'F'
        
        # Place players on the grid
        no_player = True
        for player in self.players.values():
            if not player.alive:
                continue
            
            # Mark the snake's trail
            for idx, (x, y) in enumerate(player.trail):
                #print(f"idx {idx}")
                #print(f"len(player.trail) - 1 {len(player.trail) - 1}")
                if idx == len(player.trail) - 1:
                    # print("YO!?")
                    # Snake's head
                    no_player = False
                    visual_grid[y][x] = 'X'
                else:
                    # Snake's body
                    no_player = False
                    visual_grid[y][x] = 'x'

        #if no_player:
        #    print(self.players["0"].trail)
        #    print(self.players["0"].trail)
        #    print(self.grid)
        #    exit()
        
        # Print the grid to the terminal
        print("\n" + "-" * (self.width + 2))  # Top border
        for row in visual_grid:
            print("|" + "".join(row) + "|")  # Row with side borders
        print("-" * (self.width + 2))  # Bottom border

    def visual_grid(self):
        """
        Renders the current game state to the terminal.

        :param game: The GameEnvironment instance to render.
        """
        # Initialize a 2D array representing the visual grid
        visual_grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Place food on the grid
        food_x, food_y = self.food
        visual_grid[food_x][food_y] = 'F'
        
        # Place players on the grid
        for player in self.players.values():
            if not player.alive:
                continue

            # Mark the snake's trail
            for idx, (x, y) in enumerate(player.trail):
                if idx == len(player.trail) - 1:
                    # Snake's head
                    visual_grid[x][y] = 'X'
                else:
                    # Snake's body
                    visual_grid[x][y] = 'x'
        
        return visual_grid
