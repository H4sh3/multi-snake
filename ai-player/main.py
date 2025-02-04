import asyncio
import socketio
import numpy as np
import time
from stable_baselines3 import PPO

model_name = './best_model_211_ts_18416000'

model = PPO.load(model_name, device="cpu")

obs = 7

def predict(state, player_sid):
    player = state["players"][player_sid]
    grid_size = state["gridSize"]

    food = state["food"]
    food_x = food[0]
    food_y = food[1]

    grid = np.zeros((obs, obs, 3), dtype=np.uint8)

    center = player["trail"][0]
    snake_x = center[0]
    snake_y = center[1]
    half = 3
    for xOff in range(-half,half+1,1):
        for yOff in range(-half,half+1,1):
            x = center[0] + xOff
            y = center[1] + yOff

            if x > grid_size[0]:
                grid[half+xOff, half+yOff] = [255,255,255]

            if x < 0:
                grid[half+xOff, half+yOff] = [255,255,255]

            if y > grid_size[1]:
                grid[half+xOff, half+yOff] = [255,255,255]

            if y < 0:
                grid[half+xOff, half+yOff] = [255,255,255]

            if [x,y] in player["trail"]:
                grid[half+xOff, half+yOff] = [0, 255, 0]

            if x == food_x and y == food_y:
                grid[half+xOff, half+yOff] = [255, 0, 0]


    food_direction = None
    if food_x >= snake_x and food_y >= snake_y:
        food_direction = np.array([1,0,0,0], dtype=int)
    elif food_x <= snake_x and food_y >= snake_y:
        food_direction = np.array([0,1,0,0], dtype=int)
    elif food_x >= snake_x and food_y <= snake_y:
        food_direction = np.array([0,0,1,0], dtype=int)
    elif food_x <= snake_x and food_y <= snake_y:
        food_direction = np.array([0,0,0,1], dtype=int)
    else:
        food_direction = [0,0,0,0]

    nn_input = {
        "grid": grid,
        "food_direction": food_direction
    }
    # Todo: get observation should not be implemented twice -> use env here to? maybe some interface thingy

    action, _ = model.predict(nn_input)
    
    return action

async def main():
    async with socketio.AsyncSimpleClient() as sio:
        sio = socketio.AsyncSimpleClient()
        await sio.connect("http://localhost:8000")
        print("Connected!")
        player_sid = None

        data = await sio.receive()
        if data[0] == "player_sid":
            player_sid = data[1]
            print(f"player_sid {player_sid}")

        await sio.emit("player_ready")

        for i in range(5000):

            data = await sio.receive()
            print(data)
            
            if data[0] == "state_update":

                client_time = int(time.time() * 1000)
                server_time = data[1]["t"]
                print(f"client time: {client_time}")
                print(f"server time: {server_time}")
                print(f"time delta: {client_time-server_time}")

                state = data[1]
                action = predict(state,player_sid)

                direction = None
                if action == 0: # x-1
                    direction = "left"
                elif action == 1: # y+1
                    direction = "down"
                elif action == 2: # x+1
                    direction = "right"
                elif action == 3: # y-1
                    direction = "up"
                else:
                    print("NOT GOOD")
                
                print(f"sending {direction}")
                await sio.emit("key_pressed",direction)
                print("send")



asyncio.run(main())