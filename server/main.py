from contextlib import asynccontextmanager
from fastapi import FastAPI
from socketio import AsyncServer, ASGIApp
import asyncio
from environment import GameEnvironment
from backend.player import Direction

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(game_loop())
    yield

app = FastAPI(lifespan=lifespan)
sio = AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = ASGIApp(sio)
app.mount('/', socket_app)

game_env = GameEnvironment()
TICK_RATE = 10

async def game_loop():
    while True:
        if game_env.game_active:
            game_running, winner_sid = game_env.step()
            await sio.emit('state_update', game_env.to_dict())
            if not game_running:
                game_env.game_active = False
                await sio.emit('round_end', winner_sid)
                game_env.reset()
        await asyncio.sleep(1 / TICK_RATE)

@sio.on('connect')
async def connect(sid, environ):
    player = game_env.add_player(sid)
    if player:
        await sio.emit('player_sid', sid, room=sid)
        await sio.emit('state_update', game_env.to_dict(), room=sid)
    else:
        await sio.emit('error', {'message': 'Room is full'}, room=sid)

@sio.on('player_ready')
async def player_ready(sid):
    if sid in game_env.players:
        game_env.players[sid].ready = True
        if all(p.ready for p in game_env.players.values() if p.alive):
            game_env.game_active = True
    await sio.emit('state_update', game_env.to_dict())

@sio.on('key_pressed')
async def turn(sid, direction):
    if direction == 'up':
        game_env.turn_player(sid, Direction.UP)
    if direction == 'down':
        game_env.turn_player(sid, Direction.DOWN)
    if direction == 'left':
        game_env.turn_player(sid, Direction.LEFT)
    if direction == 'right':
        game_env.turn_player(sid, Direction.RIGHT)
    

@sio.on('disconnect')
async def disconnect(sid):
    if sid in game_env.players:
        player = game_env.players[sid]
        for x, y in player.trail:
            game_env.grid[y][x] = None
        del game_env.players[sid]
        await sio.emit('state_update', game_env.to_dict())

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
