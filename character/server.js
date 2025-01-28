const express = require('express');
const cors = require('cors');
const app = express();
const port = 3000;

// Store game state
let gameState = {
    x: 400,
    y: 300,
    currentSprite: 'idle'
};

app.use(cors());
app.use(express.json());
app.use(express.static('public'));
app.use(express.static('./'));

// Control endpoints
app.post('/move', (req, res) => {
    const { direction } = req.body;
    const moveAmount = 10;
    
    switch(direction) {
        case 'left':
            gameState.x -= moveAmount;
            break;
        case 'right':
            gameState.x += moveAmount;
            break;
        case 'up':
            gameState.y -= moveAmount;
            break;
        case 'down':
            gameState.y += moveAmount;
            break;
    }
    
    res.json(gameState);
});

app.post('/changeSprite', (req, res) => {
    const { sprite } = req.body;
    gameState.currentSprite = sprite;
    res.json(gameState);
});

app.get('/state', (req, res) => {
    res.json(gameState);
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});