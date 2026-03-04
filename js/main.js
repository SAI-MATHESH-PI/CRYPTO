// VHADS Pro v2.1 - MAIN PRODUCTION PIPELINE
class VHADSApp {
    constructor() {
        this.results = [];
        this.currentFrame = 0;
        this.chartData = { hamming: [], predictions: [], times: [] };
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.updateSliders();
        console.log('🚀 VHADS Pro v2.1 Initialized');
    }
    
    bindEvents() {
        const analyzeBtn = document.getElementById('analyzeBtn');
        const uploadZone = document.getElementById('uploadZone');
        const videoInput = document.getElementById('videoInput');
        
        analyzeBtn.addEventListener('click', () => this.runAnalysis());
        
        // Drag & drop video upload
        uploadZone.addEventListener('click', () => videoInput.click());
        uploadZone.addEventListener('dragover', e => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });
        uploadZone.addEventListener('drop', e => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if(files.length) this.handleVideoUpload(files[0]);
        });
        videoInput.addEventListener('change', e => {
            if(e.target.files[0]) this.handleVideoUpload(e.target.files[0]);
        });
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
                e.target.classList.add('active');
                document.getElementById(e.target.dataset.tab + '-panel').classList.add('active');
            });
        });
        
        document.getElementById('exportBtn').addEventListener('click', () => {
            const csv = VHADSModel.exportCSV(this.results);
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `vhads-report-${Date.now()}.csv`;
            a.click();
        });
    }
    
    updateSliders() {
        const attackRateSlider = document.getElementById('attackRate');
        const frameCountSlider = document.getElementById('frameCount');
        
        attackRateSlider.addEventListener('input', () => {
            document.getElementById('attackRateValue').textContent = 
                (attackRateSlider.value * 100).toFixed(0) + '%';
        });
        
        frameCountSlider.addEventListener('input', () => {
            document.getElementById('frameCountValue').textContent = frameCountSlider.value;
        });
    }
    
    async handleVideoUpload(file) {
        const videoPreview = document.getElementById('videoPreview');
        const video = videoPreview;
        video.src = URL.createObjectURL(file);
        video.style.display = 'block';
        document.getElementById('analyzeBtn').disabled = false;
    }
    
    async runAnalysis() {
        const nFrames = parseInt(document.getElementById('frameCount').value);
        const attackRate = parseFloat(document.getElementById('attackRate').value);
        
        this.results = [];
        this.currentFrame = 0;
        this.chartData = { hamming: [], predictions: [], times: [] };
        
        // UI: Start processing
        document.getElementById('status').textContent = 'Processing...';
        document.getElementById('status').className = 'status processing';
        document.getElementById('analyzeBtn').disabled = true;
        
        // MAIN PIPELINE: 30 FPS realistic processing
        for(let i = 0; i < nFrames; i++) {
            const frameStart = performance.now();
            
            // 1. GENERATE REALISTIC CCTV FRAME
            const frameData = VideoProcessor.generateCCTVFrame(i);
            
            // 2. SELECT ATTACK TYPE (probability-based)
            const isAttack = Math.random() < attackRate;
            const attackType = isAttack ? VideoProcessor.selectAttackType() : 'normal';
            
            // 3. HIGHT ENCRYPTION + FAULT INJECTION
            const encrypted = HIGHTCipher.encrypt(frameData, attackType);
            
            // 4. COMPUTE HAMMING DISTANCE (block header)
            const hamming = HIGHTCipher.hammingDistance(frameData, encrypted);
            
            // 5. ML PREDICTION
            const prediction = VHADSModel.predict(hamming);
            
            // 6. STORE RESULT
            const result = {
                frame: i + 1,
                attack: attackType,
                prediction: prediction.prediction,
                hamming,
                confidence: prediction.confidence
            };
            this.results.push(result);
            
            // 7. REAL-TIME VISUALIZATION (30 FPS)
            VideoProcessor.renderFrame(document.getElementById('inputFrame'), frameData);
            VideoProcessor.renderEncryptedFrame(
                document.getElementById('outputFrame'), 
                encrypted, 
                frameData
            );
            
            // 8. UI UPDATES
            document.getElementById('frameNum').textContent = `${i + 1}/${nFrames}`;
            document.getElementById('hammingValue').textContent = hamming;
            document.getElementById('predictionValue').textContent = prediction.prediction.toUpperCase();
            document.getElementById('predictionValue').className = `prediction ${prediction.prediction}`;
            
            // 9. CHART UPDATE
            this.chartData.hamming.push(hamming);
            this.chartData.predictions.push(prediction.prediction);
            this.chartData.times.push(i + 1);
            this.updateRealtimeChart();
            
            // 10. PROGRESS METRICS
            this.updateMetrics();
            
            // 30 FPS THROTTLE (Realistic)
            while(performance.now() - frameStart < 33) {
                await new Promise(r => requestAnimationFrame(r));
            }
        }
        
        // FINALIZE: Update all tabs
        this.finalizeAnalysis();
        
        console.log('✅ Analysis complete:', this.results.length, 'frames processed');
    }
    
    updateRealtimeChart() {
        const chart = document.getElementById('realtimeChart');
        Plotly.newPlot(chart, [{
            x: this.chartData.times,
            y: this.chartData.hamming,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Hamming Distance',
            line: { color: '#3b82f6', width: 3 }
        }], {
            title: 'Real-time Hamming Distance Analysis',
            xaxis: { title: 'Frame' },
            yaxis: { title: 'Hamming Distance', range: [0, 10] }
        }, { responsive: true });
    }
    
    updateMetrics() {
        const normalCount = this.results.filter(r => r.prediction === 'normal').length;
        const attackCount = this.results.length - normalCount;
        const correct = this.results.filter(r => r.prediction === r.attack).length;
        const accuracy = (correct / this.results.length * 100).toFixed(1);
        
        document.getElementById('overallAccuracy').textContent = accuracy + '%';
        document.getElementById('normalFrames').textContent = normalCount;
        document.getElementById('attackFrames').textContent = attackCount;
    }
    
    finalizeAnalysis() {
        // Update status
        document.getElementById('status').textContent = 'Complete';
        document.getElementById('status').className = 'status ready';
        document.getElementById('analyzeBtn').disabled = false;
        
        // Update metrics tab
        this.renderMetricsTab();
        
        // Update report tab
        this.renderReportTab();
        
        // Show export button
        document.getElementById('exportBtn').style.display = 'block';
    }
    
    renderMetricsTab() {
        const report = VHADSModel.generateReport(this.results);
        
        Plotly.newPlot('confusionMatrix', [{
            type: 'heatmap',
            z: [[report.normal.precision, report.fault.precision, report.reduced.precision, report.differential.precision]],
            x: this.VHADSModel.classNames,
            y: ['Predictions'],
            colorscale: 'Viridis'
        }]);
    }
    
    renderReportTab() {
        const report = VHADSModel.generateReport(this.results);
        let reportText = 'VHADS Pro v2.1 - CLASSIFICATION REPORT\n';
        reportText += '===============================================\n\n';
        
        this.VHADSModel.classNames.forEach(cls => {
            const m = report[cls];
            reportText += `${cls.toUpperCase():<12} Precision: ${m.precision.toFixed(3)}\n`;
            reportText += `{' '.padStart(12)} Recall:    ${m.recall.toFixed(3)}\n`;
            reportText += `{' '.padStart(12)} F1-Score:  ${m.f1.toFixed(3)}\n`;
            reportText += `{' '.padStart(12)} Support:   ${m.support}\n\n`;
        });
        
        reportText += `Overall Accuracy: ${report.overall.accuracy.toFixed(3)} (${report.overall.total_frames} frames)\n`;
        
        document.getElementById('classificationReport').textContent = reportText;
    }
}

// Initialize when DOM loads
document.addEventListener('DOMContentLoaded', () => {
    window.vhadsApp = new VHADSApp();
});

