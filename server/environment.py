import random
from player import Player, Direction
import time

class GameEnvironment:
    def __init__(self, grid_size=(20,20), max_players=4):
        self.grid_size = grid_size
        self.max_players = max_players
        self.players = {}
        self.food = None
        # self.grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        self.game_active = False
        self.i = 0

    def add_player(self, sid):
        if len(self.players) >= self.max_players:
            return None
        player = Player(sid, self.grid_size)
        self.players[sid] = player
        if not self.food:
            self.spawn_food()
        return player

    def turn_player(self, sid, direction):
        if sid in self.players:
            self.players[sid].turn(direction)

    def spawn_food(self):

        occupied_by_players = []

        for p in self.players.items():
            occupied_by_players+=p[1].trail

        empty_cells = [
            (x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])
            if not (x,y) in occupied_by_players
        ]
        
        if empty_cells:
            self.food = random.choice(empty_cells)

    def step(self):
        self.i += 1
        winner_sid = None
        for sid, player in list(self.players.items()):
            if not player.alive:
                continue
            player.move()
            head_x, head_y = player.trail[0]

            # wall or body collision
            if head_x < 0 or head_x > self.grid_size[0] or \
                head_y < 0 or head_y > self.grid_size[1] or \
                    player.trail.count(player.trail[0]) == 2:
                player.alive = False
                continue
            if self.food and (head_x, head_y) == self.food:
                player.grow()
                self.food = None
                self.spawn_food()
            else:
                player.trail.pop()
        alive_players = [p for p in self.players.values() if p.alive]
        if len(alive_players) == 1:
            winner_sid = alive_players[0].sid
        elif not alive_players:
            self.game_active = False
        return self.game_active, winner_sid

    def reset(self):
        self.__init__(grid_size=self.grid_size, max_players=self.max_players)

    def to_dict(self):
        return {
            "gridSize": self.grid_size,
            "players": {sid: player.to_dict() for sid, player in self.players.items()},
            "food": self.food,
            "i": self.i,
            "t": int(time.time() * 1000)
        }
