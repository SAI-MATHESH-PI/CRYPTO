// HIGHTCipher v1.0 - TTAS.KO-12.0042 (Global)
const HIGHTCipher = {
    wk: new Uint8Array([0x5a,0x4d,0x39,0x2e,0x23,0x18,0x0d,0x02]),
    
    f0(x) {
        return ((x<<1|x>>>7) ^ (x<<2|x>>>6) ^ (x<<7|x>>>1)) & 0xFF;
    },
    
    f1(x) {
        return ((x<<3|x>>>5) ^ (x<<4|x>>>4) ^ (x<<6|x>>>2)) & 0xFF;
    },
    
    encrypt(plaintext, attackType = 'normal') {
        const x = new Uint8Array(plaintext.slice(0,8));
        
        // Input whitening
        for(let i=0; i<8; i++) {
            x[i] = (i%2===0) ? (x[i] + this.wk[i])&0xFF : x[i] ^ this.wk[i];
        }
        
        // 32 rounds
        const sk = new Uint8Array(128);
        for(let i=0; i<128; i++) sk[i] = (0x01 + i)&0xFF;
        
        for(let r=0; r<32; r++) {
            const sk4 = sk.slice(r*4, r*4+4);
            const t0 = (this.f0(x[6]) + sk4[3]) & 0xFF; x[0] = x[7] ^ t0;
            const t1 = this.f1(x[0]) ^ sk4[2]; x[2] = (x[1] + t1) & 0xFF;
            const t2 = (this.f0(x[2]) + sk4[1]) & 0xFF; x[4] = x[3] ^ t2;
            const t3 = this.f1(x[4]) ^ sk4[0]; x[6] = (x[5] + t3) & 0xFF;
            
            // Attack simulation
            if(attackType === 'fault' && r>=12 && r<=20) {
                x[Math.floor(Math.random()*8)] ^= 0xAA;
            } else if(attackType === 'reduced' && r>=24) {
                break;
            } else if(attackType === 'differential' && r%4===0) {
                x[0] ^= (r+1)&0x0F;
            }
        }
        
        // Output whitening
        const ct = new Uint8Array(8);
        for(let i=0; i<8; i++) {
            const idx = (i+1)%8;
            ct[i] = (i%2===0) ? (x[idx] + this.wk[i])&0xFF : x[idx] ^ this.wk[i];
        }
        return ct;
    }
};
