/**
 * CCTV Video Frame Processing Pipeline
 * 64x64 pixel blocks • 30 FPS real-time
 */
export class VideoProcessor {
    constructor() {
        this.blockSize = 16; // 64x64 = 4x4 blocks
        this.frameBuffer = new Uint8Array(4096); // 64*64
    }

    /**
     * Generate realistic CCTV frame (YUV-like)
     */
    generateCCTVFrame() {
        const frame = new Uint8Array(4096);
        
        // Base scene (office/security camera)
        for (let y = 0; y < 64; y++) {
            for (let x = 0; x < 64; x++) {
                let intensity = 128;
                
                // Lighting gradient
                intensity += Math.sin(x * 0.1) * 20;
                intensity += Math.cos(y * 0.08) * 15;
                
                // Moving objects (people/vehicles)
                if (x > 20 && x < 40 && y > 10 && y < 30) {
                    intensity -= Math.sin(Date.now() * 0.001 + x * 0.05) * 40;
                }
                
                frame[y * 64 + x] = Math.max(0, Math.min(255, Math.round(intensity)));
            }
        }
        
        return frame;
    }

    /**
     * Select realistic attack type based on probability
     */
    selectAttackType() {
        const attacks = [
            { type: 'fault', prob: 0.45 },
            { type: 'reduced', prob: 0.30 },
            { type: 'differential', prob: 0.25 }
        ];
        
        let rand = Math.random();
        for (const attack of attacks) {
            if (rand < attack.prob) return attack.type;
            rand -= attack.prob;
        }
        return 'fault';
    }

    /**
     * Render frame to HTML5 canvas
     */
    renderFrame(canvas, frameData) {
        const ctx = canvas.getContext('2d');
        const blockSize = this.blockSize;
        
        ctx.imageSmoothingEnabled = false;
        
        for (let y = 0; y < 64; y += 4) {
            for (let x = 0; x < 64; x += 4) {
                const idx = y * 64 + x;
                const val = frameData[idx] || 128;
                
                ctx.fillStyle = `hsl(${240 - val * 0.8}, 60%, ${val * 0.4}%)`;
                ctx.fillRect(x * blockSize / 4, y * blockSize / 4, blockSize, blockSize);
                
                // Grid lines
                ctx.strokeStyle = `rgba(31, 41, 55, ${0.1 + val * 0.002})`;
                ctx.lineWidth = 1;
                ctx.strokeRect(x * blockSize / 4, y * blockSize / 4, blockSize, blockSize);
            }
        }
    }

    /**
     * Render encrypted frame with visual fault indicators
     */
    renderEncryptedFrame(canvas, encrypted, original) {
        const ctx = canvas.getContext('2d');
        const blockSize = this.blockSize;
        
        ctx.imageSmoothingEnabled = false;
        
        for (let y = 0; y < 64; y += 4) {
            for (let x = 0; x < 64; x += 4) {
                const idx = y * 64 + x;
                const origVal = original[idx] || 128;
                const encVal = encrypted[Math.floor(idx / 256)] || 128;
                
                // Encrypted visualization (ciphertext texture)
                const hue = (encVal * 1.2) % 360;
                ctx.fillStyle = `hsl(${hue}, 70%, ${35 + encVal * 0.3}%)`;
                ctx.fillRect(x * blockSize / 4, y * blockSize / 4, blockSize, blockSize);
                
                // Fault indicators
                const diff = Math.abs(origVal - encVal);
                if (diff > 32) {
                    ctx.strokeStyle = '#ef4444';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x * blockSize / 4, y * blockSize / 4, blockSize, blockSize);
                } else {
                    ctx.strokeStyle = `rgba(31, 41, 55, ${0.15 + encVal * 0.001})`;
                    ctx.lineWidth = 1;
                    ctx.strokeRect(x * blockSize / 4, y * blockSize / 4, blockSize, blockSize);
                }
            }
        }
    }

    /**
     * Calculate block-level Hamming distance
     */
    calculateHamming(original, encrypted) {
        let distance = 0;
        for (let i = 0; i < Math.min(original.length, encrypted.length); i++) {
            distance += (original[i] ^ encrypted[i]) !== 0 ? 1 : 0;
        }
        return distance;
    }
}
