const config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    scene: {
        preload: preload,
        create: create,
        update: update
    }
};

let game;
window.onload = () => {
    game = new Phaser.Game(config);
};

let character;
let lastUpdate = 0;
let idleOffset = 0;
let idleDirection = 1;

function preload() {
    this.load.spritesheet('character', 'character.png', { 
        frameWidth: 434, 
        frameHeight: 515 
    });
}

function create() {
    character = this.add.sprite(434, 515, 'character');
    
    // Set up animations
    this.anims.create({
        key: 'idle',
        frames: this.anims.generateFrameNumbers('character', { start: 0, end: 3 }),
        frameRate: 8,
        repeat: -1
    });
    
    character.play('idle');
}

function update(time) {
    const now = Date.now();
    
    // Add subtle idle movement
    idleOffset += 0.05 * idleDirection;
    if (Math.abs(idleOffset) >= 1) {
        idleDirection *= -1;  // Reverse direction
    }
    
    // Apply a gentle up/down movement
    character.y += Math.sin(time / 500) * 0.15;  // Subtle breathing motion
    
    // Apply a very slight side-to-side sway
    character.x += Math.sin(time / 1000) * 0.1;
    
    // Add a barely noticeable scale bounce
    const scaleOffset = Math.sin(time / 400) * 0.001;
    character.setScale(1 + scaleOffset, 1 - scaleOffset);
    
    // Poll server for updates every 100ms
    if (now - lastUpdate > 100) {
        fetch('http://localhost:3000/state')
            .then(response => response.json())
            .then(state => {
                // Use the server position as the center point for idle animations
                character.x = state.x + idleOffset;
                
                if (character.anims.currentAnim.key !== state.currentSprite) {
                    character.play(state.currentSprite);
                }
            });
        
        lastUpdate = now;
    }
}