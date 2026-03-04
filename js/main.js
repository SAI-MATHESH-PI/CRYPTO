// VHADS Pro v2.0 Main Application (Global)
document.addEventListener('DOMContentLoaded', function() {
    let state = { results: [], isAnalyzing: false };
    
    // Event bindings
    document.getElementById('videoInput').addEventListener('change', e => {
        const file = e.target.files[0];
        if(file && file.type.startsWith('video/')) {
            const url = URL.createObjectURL(file);
            document.getElementById('videoPreview').innerHTML = 
                `<video src="${url}" controls style="max-height:200px;"></video>`;
            document.getElementById('analyzeBtn').disabled = false;
        }
    });
    
    document.getElementById('analyzeBtn').addEventListener('click', runAnalysis);
    
    ['attackRate','frameCount'].forEach(id => {
        document.getElementById(id).addEventListener('input', e => {
            document.getElementById(id+'Value').textContent = e.target.value;
        });
    });
    
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab+'-panel').classList.add('active');
        });
    });
    
    Charts.init();
    
    async function runAnalysis() {
        state.isAnalyzing = true;
        document.getElementById('analyzeBtn').disabled = true;
        
        const nFrames = parseInt(document.getElementById('frameCount').value);
        state.results = [];
        
        const inputCanvas = document.getElementById('inputFrame');
        const outputCanvas = document.getElementById('outputFrame');
        
        for(let i=0; i<nFrames; i++) {
            const frameData = VideoProcessor.generateCCTVFrame();
            const attackType = Math.random()<0.25 ? 
                VideoProcessor.selectAttackType() : 'normal';
                
            const encrypted = HIGHTCipher.encrypt(frameData, attackType);
            const hamming = frameData.slice(0,8).filter((b,j)=>b!==encrypted[j]).length;
            const pred = VHADSModel.predict(hamming, attackType);
            
            state.results.push({frame:i+1,attack:attackType,prediction:pred.prediction,hamming,confidence:pred.confidence});
            
            VideoProcessor.renderFrame(inputCanvas, frameData);
            VideoProcessor.renderEncryptedFrame(outputCanvas, encrypted, frameData);
            
            document.getElementById('frameNum').textContent = i+1;
            document.getElementById('hammingValue').textContent = hamming;
            document.getElementById('predictionValue').textContent = pred.prediction.toUpperCase();
            
            await new Promise(r=>setTimeout(r,30));
        }
        
        // Update metrics
        const normalCount = state.results.filter(r=>r.prediction==='normal').length;
        document.getElementById('overallAccuracy').textContent = (normalCount/nFrames*100).toFixed(1)+'%';
        document.getElementById('normalFrames').textContent = normalCount;
        document.getElementById('attackFrames').textContent = nFrames-normalCount;
        
        Charts.updateRealtime(state.results);
        renderReport();
        
        state.isAnalyzing = false;
        document.getElementById('analyzeBtn').disabled = false;
    }
    
    function renderReport() {
        const report = VHADSModel.generateReport(state.results);
        document.getElementById('classificationReport').innerHTML = `
            <table class="results-table">
                <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
                ${Object.entries(report).map(([k,v])=>
                    `<tr><td>${k}</td><td>${v.precision.toFixed(3)}</td><td>${v.recall.toFixed(3)}</td><td>${v.f1.toFixed(3)}</td></tr>`
                ).join('')}
            </table>
        `;
    }
});

