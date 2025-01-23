import { useEffect, useRef, useState } from 'react';
import { io } from 'socket.io-client';

interface Player {
  x: number;
  y: number;
  direction: string;
  color: string;
  alive: boolean;
  ready: boolean;
  wins: number;
  trail: [number, number][]; // Added trail property for rendering
}

interface GameState {
  gridSize: [number, number]; // width height
  players: Record<string, Player>;
  food: [number, number]; // Added food position
  active: boolean;
}

// const socket = io('http://192.168.178.27:8000', {
const socket = io('http://127.0.0.1:8000', {
  autoConnect: false,
});

const Game = () => {
  const [uid, setUid] = useState<string>('');
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [winnerSid, setWinnerSid] = useState<string | null>(null); // Track winner SID for round end
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    socket.on('state_update', (newState: GameState) => {
      setGameState(newState);
    });

    socket.on('round_end', (winnerSid: string | null) => {
      setWinnerSid(winnerSid);
      // alert(winnerSid ? `Player ${winnerSid} won the round!` : 'No winner this round (tie).');
    });

    socket.on('player_sid', (sid: string) => {
      setUid(sid);
    });

    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        socket.emit('key_pressed', 'left');
      } else if (e.key === 'ArrowRight') {
        socket.emit('key_pressed', 'right');
      } else if (e.key === 'ArrowUp') {
        socket.emit('key_pressed', 'up');
      } else if (e.key === 'ArrowDown') {
        socket.emit('key_pressed', 'down');
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    socket.connect();

    return () => {
      window.removeEventListener('keydown', handleKeyPress);
      socket.disconnect();
    };
  }, []);

  useEffect(() => {
    if (!gameState || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cellSize = 10;
    canvas.width = gameState.gridSize[0] * cellSize;
    canvas.height = gameState.gridSize[1] * cellSize;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw food
    const [foodX, foodY] = gameState.food;
    ctx.fillStyle = '#00FF00'; // White color for food
    ctx.fillRect(foodX * cellSize, foodY * cellSize, cellSize, cellSize);

    // Draw players and their trails
    Object.values(gameState.players).forEach((player) => {
      // Draw the player's trail
      player.trail.forEach(([trailX, trailY]) => {

        ctx.fillStyle = '#0000FF'; // White color for food
        // ctx.fillStyle = player.color;
        ctx.fillRect(trailX * cellSize, trailY * cellSize, cellSize, cellSize);
      });

      // Highlight the player's head
      ctx.fillStyle = player.color;
      ctx.fillRect(player.x * cellSize, player.y * cellSize, cellSize, cellSize);
    });
  }, [gameState]);

  const playerReady = () => {
    socket.emit('player_ready');
  };

  if (!gameState) {
    return <div className="text-center p-4">Connecting...</div>;
  }

  return (
    <div className="flex flex-col items-center gap-4 p-4">
      {Object.entries(gameState.players).map(([sid, player]) => (
        <div key={sid}>
          {sid === uid ? '-> ' : ''}
          Ready: {player.ready ? 'Yes' : 'No'} | Wins: {player.wins}
        </div>
      ))}
      <div className="flex gap-4 mb-4">
        <div>Players: {Object.keys(gameState.players).length}/4</div>
        {!gameState.active && (
          <button
            onClick={playerReady}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Ready
          </button>
        )}
      </div>
      <canvas ref={canvasRef} className="border border-gray-300" />
      <div className="text-sm text-gray-600">
        Use Left/Right arrows to turn
      </div>
    </div>
  );
};

export default Game;
