from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from socketio import AsyncServer, ASGIApp
import asyncio
from enum import Enum
from typing import Dict, List, Tuple

class Direction(Enum):
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)

class Player:
    def __init__(self, x: int, y: int, direction: Direction, color: str):
        self.x = x
        self.y = y
        self.direction = direction
        self.color = color
        self.trail: List[Tuple[int, int]] = [(x, y)]
        self.alive = True
        self.ready = False
        self.sid = None
        self.wins = 0

class GameState:
    def __init__(self, width: int = 50, height: int = 50):
        self.width = width
        self.height = height
        self.players: Dict[str, Player] = {}
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.food: Tuple[int, int] = self.spawn_food()
        self.game_active = False

    def spawn_food(self) -> Tuple[int, int]:
        """Spawn food in a random unoccupied position on the grid."""
        import random
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.grid[y][x] is None:  # Ensure food is not on a player or trail
                return x, y

    def add_player(self, sid: str) -> Player:
        if len(self.players) >= 4:  # Limit to 4 players
            return None
            
        # Calculate spawn position based on player count
        positions = [
            (5, 5, Direction.RIGHT, "#FF0000"),
            (self.width - 5, self.height - 5, Direction.LEFT, "#0000FF"),
            (5, self.height - 5, Direction.RIGHT, "#00FF00"),
            (self.width - 5, 5, Direction.LEFT, "#FFFF00")
        ]
        pos = positions[len(self.players)]
        
        player = Player(*pos)
        player.sid = sid
        player.trail = [pos[:2]] * 5  # Start with a tail length of 5
        self.players[sid] = player
        self.grid[player.y][player.x] = player.color
        return player

    def update(self) -> Tuple[bool, str]:
        alive_players = [p for p in self.players.values() if p.alive]
        if len(alive_players) == 1 and len(self.players.values()) > 1:
            alive_players[0].wins += 1
            return False, alive_players[0].sid  # Winner's color
        if len(alive_players) == 0:
            return False, None  # No winner (tie)
        
        for player in alive_players:
            dx, dy = player.direction.value
            new_x = (player.x + dx) % self.width  # Wrap around horizontally
            new_y = (player.y + dy) % self.height  # Wrap around vertically

            # Check collision with trails
            if self.grid[new_y][new_x] is not None:
                # Player dies
                player.alive = False
                # Remove player trail from the grid
                for x, y in player.trail:
                    self.grid[y][x] = None
                continue

            # Check for food pickup
            if (new_x, new_y) == self.food:
                player.trail.append((new_x, new_y))  # Extend tail by 1
                self.food = self.spawn_food()  # Spawn new food
            else:
                # Remove the oldest tail segment
                tail_end = player.trail.pop(0)
                self.grid[tail_end[1]][tail_end[0]] = None

            # Update position
            player.x = new_x
            player.y = new_y
            player.trail.append((new_x, new_y))
            self.grid[new_y][new_x] = player.color
            
        return True, None  # Game still active

    def turn_player(self, player_id: str, direction: str):
        if player_id not in self.players:
            print("no player found")
            return

        player = self.players[player_id]
        if not player.alive:
            print("player is dead")
            return

        # Map input strings to Direction enums
        input_to_direction = {
            "up": Direction.UP,
            "right": Direction.RIGHT,
            "down": Direction.DOWN,
            "left": Direction.LEFT
        }

        new_direction = input_to_direction.get(direction)
        print(new_direction)
        if not new_direction:
            return  # Invalid direction input

        current = player.direction
        # Prevent turning into the opposite direction
        if new_direction.value[0] + current.value[0] == 0 and new_direction.value[1] + current.value[1] == 0:
            return

        player.direction = new_direction


    def reset(self):
        """Resets the game state for a new round."""
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
            player.trail = [(x, y)] * 5  # Reset tail length to 5
            self.grid[y][x] = color
        self.food = self.spawn_food()
        self.game_active = False

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



@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(game_loop())
    yield
    print("done!")

app = FastAPI(lifespan=lifespan)
sio = AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = ASGIApp(sio)
app.mount('/', socket_app)

game_state = GameState()
TICK_RATE = 10  # Game updates per second

async def game_loop():
    while True:
        if game_state.game_active:
            game_running, winner_sid = game_state.update()
            await sio.emit('state_update', game_state.to_dict())
            if not game_running:
                game_state.game_active = False
                if winner_sid:
                    print(f"Player with sid {winner_sid} has won!")
                else:
                    print("No winner this round (tie).")
                await sio.emit('round_end', winner_sid)
                game_state.reset()
        await asyncio.sleep(1 / TICK_RATE)

@sio.on('connect')
async def connect(sid, environ):
    print(f"New player connected: {sid}")
    player = game_state.add_player(sid)
    if player:
        await sio.emit('player_sid', sid, room=sid)
        await sio.emit('state_update', game_state.to_dict(), room=sid)
    else:
        await sio.emit('error', {'message': 'Room is full'}, room=sid)

@sio.on('player_ready')
async def player_ready(sid):
    if sid in game_state.players:
        game_state.players[sid].ready = True
        if all(p.ready for p in game_state.players.values() if p.alive):
            game_state.game_active = True
            print("All players ready. Starting game!")
    await sio.emit('state_update', game_state.to_dict())

@sio.on('key_pressed')
async def turn(sid, direction):
    game_state.turn_player(sid, direction)

@sio.on('disconnect')
async def disconnect(sid):
    if sid in game_state.players:
        player = game_state.players[sid]
        for x, y in player.trail:
            game_state.grid[y][x] = None
        del game_state.players[sid]
        await sio.emit('state_update', game_state.to_dict())

#@app.on_event("startup")
#async def startup_event():
#    asyncio.create_task(game_loop())

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
