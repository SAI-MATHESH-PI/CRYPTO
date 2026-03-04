// VideoProcessor v1.0 - CCTV Frame Generation (Global)
const VideoProcessor = {
    generateCCTVFrame() {
        const frame = new Uint8Array(64*64);
        for(let y=0; y<64; y++) {
            for(let x=0; x<64; x++) {
                let val = 128;
                val += Math.sin(x*0.1)*25;
                val += Math.cos(y*0.08)*20;
                if(x>20&&x<40&&y>10&&y<30) {
                    val -= Math.sin(Date.now()*0.001 + x*0.05)*45;
                }
                frame[y*64+x] = Math.max(0,Math.min(255,Math.round(val)));
            }
        }
        return frame;
    },
    
    selectAttackType() {
        const r = Math.random();
        return r<0.45 ? 'fault' : r<0.75 ? 'reduced' : 'differential';
    },
    
    renderFrame(canvas, frameData) {
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        for(let y=0; y<64; y+=4) {
            for(let x=0; x<64; x+=4) {
                const idx = y*64 + x;
                const val = frameData[idx]||128;
                ctx.fillStyle = `hsl(240,60%,${val*0.4}%)`;
                ctx.fillRect(x*6.25,y*4.7,25,22.5);
                ctx.strokeStyle = 'rgba(0,0,0,0.1)';
                ctx.lineWidth = 1;
                ctx.strokeRect(x*6.25,y*4.7,25,22.5);
            }
        }
    },
    
    renderEncryptedFrame(canvas, encrypted, original) {
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        for(let y=0; y<64; y+=4) {
            for(let x=0; x<64; x+=4) {
                const idx = y*64 + x;
                const orig = original[idx]||128;
                const enc = encrypted[Math.floor(idx/256)]||128;
                ctx.fillStyle = `hsl(${(enc*1.2)%360},70%,${35+enc*0.3}%)`;
                ctx.fillRect(x*6.25,y*4.7,25,22.5);
                const diff = Math.abs(orig-enc);
                ctx.strokeStyle = diff>32 ? '#ef4444' : 'rgba(0,0,0,0.15)';
                ctx.lineWidth = diff>32 ? 3 : 1;
                ctx.strokeRect(x*6.25,y*4.7,25,22.5);
            }
        }
    }
};
