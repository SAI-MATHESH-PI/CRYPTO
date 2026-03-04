// VideoProcessor v2.0 - Realistic CCTV Frame Generation & Processing
const VideoProcessor = {
    // CCTV frame dimensions (64x64 = 4096 pixels)
    blockSize: 16,
    frameWidth: 64,
    frameHeight: 64,
    
    // Generate realistic CCTV frame (YUV-like grayscale)
    generateCCTVFrame(timestamp = Date.now()) {
        const frame = new Uint8Array(this.frameWidth * this.frameHeight);
        const t = timestamp * 0.001; // Animation time
        
        for (let y = 0; y < this.frameHeight; y++) {
            for (let x = 0; x < this.frameWidth; x++) {
                let intensity = 128; // Base lighting (50% gray)
                
                // Realistic office/security camera gradients
                intensity += Math.sin(x * 0.12) * 30;      // Horizontal lighting
                intensity += Math.cos(y * 0.09) * 25;      // Vertical lighting
                intensity += Math.sin((x + y) * 0.08) * 15; // Texture
                
                // Moving objects simulation (people/vehicles)
                if (x > 18 && x < 42 && y > 12 && y < 28) {
                    const motion = Math.sin(t * 2 + x * 0.06 + y * 0.04) * 55;
                    intensity -= motion; // Shadow/movement
                }
                
                // Timestamp flicker (CCTV timestamp overlay effect)
                if (x > 2 && x < 12 && y > 2 && y < 6) {
                    intensity += Math.sin(t * 5 + x * y) * 20;
                }
                
                frame[y * this.frameWidth + x] = Math.max(0, Math.min(255, Math.round(intensity)));
            }
        }
        
        return frame;
    },
    
    // Select attack type based on realistic probabilities
    selectAttackType() {
        const attackTypes = [
            { type: 'fault', probability: 0.45 },      // Most common (transient faults)
            { type: 'reduced', probability: 0.30 },    // Reduced-round attacks
            { type: 'differential', probability: 0.25 } // Differential fault analysis
        ];
        
        let rand = Math.random();
        for (const attack of attackTypes) {
            if (rand < attack.probability) {
                return attack.type;
            }
            rand -= attack.probability;
        }
        return 'fault'; // Default
    },
    
    // Render frame to HTML5 Canvas (pixel block visualization)
    renderFrame(canvas, frameData) {
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        
        // 16x16 pixel blocks (4x4 grid of 16x16 blocks)
        for (let blockY = 0; blockY < 16; blockY++) {
            for (let blockX = 0; blockX < 16; blockX++) {
                // Sample center pixel of each block
                const pixelX = blockX * 4;
                const pixelY = blockY * 4;
                const idx = pixelY * 64 + pixelX;
                const val = frameData[idx] || 128;
                
                // Grayscale to HSL for realistic CCTV look
                ctx.fillStyle = `hsl(0, 0%, ${val * 0.45}%)`;
                ctx.fillRect(blockX * 25, blockY * 18.75, 25, 18.75);
                
                // Grid lines
                ctx.strokeStyle = `rgba(31, 41, 55, ${0.12 + val * 0.002})`;
                ctx.lineWidth = 1;
                ctx.strokeRect(blockX * 25, blockY * 18.75, 25, 18.75);
            }
        }
    },
    
    // Render encrypted frame with fault visualization
    renderEncryptedFrame(canvas, encryptedData, originalFrame) {
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        
        for (let blockY = 0; blockY < 16; blockY++) {
            for (let blockX = 0; blockX < 16; blockX++) {
                const pixelX = blockX * 4;
                const pixelY = blockY * 4;
                const frameIdx = pixelY * 64 + pixelX;
                
                const origVal = originalFrame[frameIdx] || 128;
                const encVal = encryptedData[Math.floor(frameIdx / 64)] || 128;
                
                // Ciphertext visualization (psychedelic encryption effect)
                const hue = (encVal * 1.3 + Date.now() * 0.001) % 360;
                ctx.fillStyle = `hsl(${hue}, 75%, ${30 + encVal * 0.35}%)`;
                ctx.fillRect(blockX * 25, blockY * 18.75, 25, 18.75);
                
                // Fault indicators (red borders for high differences)
                const diff = Math.abs(origVal - encVal);
                if (diff > 35) {
                    ctx.strokeStyle = '#ef4444';
                    ctx.lineWidth = 3;
                } else {
                    ctx.strokeStyle = `rgba(31, 41, 55, ${0.2 + encVal * 0.001})`;
                    ctx.lineWidth = 1;
                }
                ctx.strokeRect(blockX * 25, blockY * 18.75, 25, 18.75);
            }
        }
    },
    
    // Calculate block-wise Hamming distance
    calculateHamming(original, encrypted) {
        let distance = 0;
        const minLen = Math.min(original.length, encrypted.length);
        for (let i = 0; i < minLen; i++) {
            distance += (original[i] ^ encrypted[i]) !== 0 ? 1 : 0;
        }
        return distance;
    }
};

console.log('✅ VideoProcessor loaded - 64x64 CCTV frame generation');
