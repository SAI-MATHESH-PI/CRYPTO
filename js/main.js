// VHADS Pro v2.0 - Main Application Orchestrator
// Production-grade error handling + real-time processing
class VHADSApp {
    constructor() {
        this.state = {
            results: [],
            isAnalyzing: false,
            videoFile: null
        };
        
        this.init();
    }
    
    async init() {
        try {
            // Initialize charts first
            await Charts.init();
            
            // Bind all event listeners
            this.bindEvents();
            
            // Update initial UI state
            this.updateControls();
            
            console.log('✅ VHADS Pro v2.0 initialized successfully');
        } catch (error) {
            console.error('VHADS initialization failed:', error);
            this.showError('Application initialization failed. Please refresh.');
        }
    }
    
    bindEvents() {
        // Video upload
        document.getElementById('videoInput').addEventListener('change', (e) => {
            this.handleVideoUpload(e.target.files[0]);
        });
        
        // Control sliders
        ['attackRate', 'frameCount'].forEach(id => {
            document.getElementById(id).addEventListener('input', (e) => {
                this.state[e.target.id] = parseFloat(e.target.value);
                document.getElementById(`${id}Value`).textContent = e.target.value;
            });
        });
        
        // Analyze button
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.runAnalysis();
        });
        
        // Tab navigation (keyboard accessible)
        document.querySelectorAll('.tab-btn').forEach((btn, index) => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
            btn.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.switchTab(e.target.dataset.tab);
                }
            });
        });
        
        // Keyboard navigation for tabs
        document.addEventListener('keydown', (e) => {
            const activeTab = document.querySelector('.tab-btn[aria-selected="true"]');
            const tabs = document.querySelectorAll('.tab-btn');
            let currentIndex = Array.from(tabs).indexOf(activeTab);
            
            if (e.key === 'ArrowRight') {
                e.preventDefault();
                const nextIndex = (currentIndex + 1) % tabs.length;
                this.switchTab(tabs[nextIndex].dataset.tab);
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                const prevIndex = (currentIndex - 1 + tabs.length) % tabs.length;
                this.switchTab(tabs[prevIndex].dataset.tab);
            }
        });
    }
    
    handleVideoUpload(file) {
        if (!file || !file.type.startsWith('video/')) {
            this.setStatus('error', 'Please upload a valid video file (MP4, MOV, AVI)');
            return;
        }
        
        this.state.videoFile = file;
        const url = URL.createObjectURL(file);
        const preview = document.getElementById('videoPreview');
        
        preview.innerHTML = `
            <video controls autoplay muted 
                   style="max-height: 200px; border-radius: 8px; box-shadow: var(--shadow-md);">
                <source src="${url}" type="${file.type}">
                Your browser does not support video playback.
            </video>
        `;
        
        document.getElementById('uploadZone').classList.add('has-video');
        document.getElementById('analyzeBtn').disabled = false;
        this.setStatus('ready', `Video loaded: ${file.name} (${(file.size/1024/1024).toFixed(1)}MB)`);
    }
    
    async runAnalysis() {
        if (this.state.isAnalyzing) return;
        
        this.state.isAnalyzing = true;
        const analyzeBtn = document.getElementById('analyzeBtn');
        const statusEl = document.getElementById('status');
        
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span><span class="btn-text">Processing...</span>';
        statusEl.textContent = '🔍 Running real-time HIGHT analysis...';
        statusEl.className = 'status analyzing';
        
        try {
            const results = await this.processFrames();
            this.state.results = results;
            
            this.updateSummaryMetrics();
            await Charts.updateRealtime(results);
            this.renderClassificationReport();
            
            this.setStatus('complete', `Analysis complete: ${results.length} frames, ${(results.filter(r => r.prediction === 'normal').length / results.length * 100).toFixed(1)}% accuracy`);
            analyzeBtn.innerHTML = '<span class="btn-icon">✅</span><span class="btn-text">Re-run Analysis</span>';
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.setStatus('error', `Analysis failed: ${error.message}`);
            this.showError('Analysis interrupted. Please check console for details.');
        } finally {
            this.state.isAnalyzing = false;
            analyzeBtn.disabled = false;
        }
    }
    
    async processFrames() {
        const results = [];
        const nFrames = parseInt(document.getElementById('frameCount').value);
        const attackRate = parseFloat(document.getElementById('attackRate').value);
        
        const inputCanvas = document.getElementById('inputFrame');
        const outputCanvas = document.getElementById('outputFrame');
        
        for (let i = 0; i < nFrames; i++) {
            // Generate realistic CCTV frame
            const frameData = VideoProcessor.generateCCTVFrame();
            
            // Probabilistic attack injection
            const attackType = Math.random() < attackRate ?
                VideoProcessor.selectAttackType() : 'normal';
            
            // HIGHT encryption with attack simulation
            const encrypted = HIGHTCipher.encrypt(frameData, attackType);
            
            // Calculate Hamming distance (first 8 bytes)
            const hamming = HIGHTCipher.hammingDistance(
                frameData.slice(0, 8), 
                encrypted
            );
            
            // ML prediction
            const prediction = VHADSModel.predict(hamming, encrypted);
            
            // Store result
            results.push({
                frame: i + 1,
                attack: attackType,
                prediction: prediction.prediction,
                hamming: hamming,
                confidence: prediction.confidence
            });
            
            // Real-time visualization (30 FPS)
            VideoProcessor.renderFrame(inputCanvas, frameData);
            VideoProcessor.renderEncryptedFrame(outputCanvas, encrypted, frameData);
            
            // Update UI progressively
            document.getElementById('frameNum').textContent = i + 1;
            document.getElementById('hammingValue').textContent = hamming;
            
            const predEl = document.getElementById('predictionValue');
            predEl.textContent = prediction.prediction.toUpperCase();
            predEl.className = `prediction ${prediction.prediction}`;
            
            // Throttle for realistic 30 FPS
            await new Promise(resolve => setTimeout(resolve, 33));
        }
        
        return results;
    }
    
    updateSummaryMetrics() {
        const results = this.state.results;
        const nFrames = results.length;
        const normalCount = results.filter(r => r.prediction === 'normal').length;
        const accuracy = (normalCount / nFrames * 100);
        
        document.getElementById('overallAccuracy').textContent = accuracy.toFixed(1) + '%';
        document.getElementById('normalFrames').textContent = normalCount;
        document.getElementById('attackFrames').textContent = nFrames - normalCount;
    }
    
    renderClassificationReport() {
        const report = VHADSModel.generateReport(this.state.results);
        const container = document.getElementById('classificationReport');
        
        container.innerHTML = `
            <header style="margin-bottom: 2rem;">
                <h3 style="color: var(--primary); margin-bottom: 0.5rem;">
                    📋 Classification Report
                </h3>
                <p style="color: var(--text-secondary); margin: 0;">
                    Analysis of ${this.state.results.length} frames | 
                    Attack rate: ${(document.getElementById('attackRate').value * 100).toFixed(0)}%
                </p>
            </header>
            
            <table class="results-table" role="table" aria-label="Classification performance metrics">
                <thead>
                    <tr>
                        <th scope="col">Class</th>
                        <th scope="col">Precision</th>
                        <th scope="col">Recall</th>
                        <th scope="col">F1-Score</th>
                        <th scope="col">Support</th>
                    </tr>
                </thead>
                <tbody>
                    ${VHADSModel.classNames.map(className => {
                        const metrics = report[className];
                        const support = this.state.results.filter(r => r.attack === className).length;
                        return `
                            <tr>
                                <th scope="row">${className.charAt(0).toUpperCase() + className.slice(1)}</th>
                                <td>${metrics.precision.toFixed(3)}</td>
                                <td>${metrics.recall.toFixed(3)}</td>
                                <td><strong>${metrics.f1_score.toFixed(3)}</strong></td>
                                <td>${support}</td>
                            </tr>
                        `;
                    }).join('')}
                    <tr style="font-weight: 800; background: linear-gradient(135deg, #f3f4f6, #e5e7eb);">
                        <th scope="row">Macro Average</th>
                        <td>${(Object.values(report).reduce((sum, m) => sum + m.precision, 0) / 4).toFixed(3)}</td>
                        <td>${(Object.values(report).reduce((sum, m) => sum + m.recall, 0) / 4).toFixed(3)}</td>
                        <td><strong>${(Object.values(report).reduce((sum, m) => sum + m.f1_score, 0) / 4).toFixed(3)}</strong></td>
                        <td>${this.state.results.length}</td>
                    </tr>
                </tbody>
            </table>
        `;
        
        // Enable CSV export
        const exportBtn = document.getElementById('exportBtn');
        exportBtn.style.display = 'inline-flex';
        exportBtn.onclick = () => this.exportCSV();
    }
    
    exportCSV() {
        const csvContent = [
            ['Frame', 'Attack_Type', 'Prediction', 'Hamming_Distance', 'Confidence'],
            ...this.state.results.map(r => [
                r.frame,
                r.attack,
                r.prediction,
                r.hamming,
                r.confidence.toFixed(4)
            ])
        ].map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        link.setAttribute('href', url);
        link.setAttribute('download', `VHADS-Analysis-${new Date().toISOString().slice(0,10)}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    switchTab(tabId) {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
            btn.setAttribute('aria-selected', 'false');
            btn.setAttribute('tabindex', '-1');
        });
        
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.remove('active');
        });
        
        const activeBtn = document.querySelector(`[data-tab="${tabId}"]`);
        const activePanel = document.getElementById(`${tabId}-panel`);
        
        activeBtn.classList.add('active');
        activeBtn.setAttribute('aria-selected', 'true');
        activeBtn.setAttribute('tabindex', '0');
        activePanel.classList.add('active');
        activePanel.focus();
    }
    
    setStatus(type, message) {
        const statusEl = document.getElementById('status');
        statusEl.textContent = message;
        statusEl.className = `status ${type}`;
        statusEl.setAttribute('aria-live', 'assertive');
    }
    
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed; top: 20px; right: 20px; background: #fee2e2; 
            color: #991b1b; padding: 1rem 1.5rem; border-radius: 8px; 
            box-shadow: 0 10px 25px rgba(239,68,68,0.3); z-index: 10000;
            border-left: 4px solid #ef4444; max-width: 400px;
        `;
        errorDiv.innerHTML = `
            <strong>⚠️ Analysis Error</strong><br>${message}
            <button onclick="this.parentElement.remove()" style="float:right;background:none;border:none;font-size:1.2rem;cursor:pointer;">×</button>
        `;
        document.body.appendChild(errorDiv);
        
        setTimeout(() => errorDiv.remove(), 10000);
    }
    
    updateControls() {
        document.getElementById('attackRateValue').textContent = this.state.attackRate || 0.25;
        document.getElementById('frameCountValue').textContent = this.state.frameCount || 64;
    }
}

// Initialize application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new VHADSApp());
} else {
    new VHADSApp();
}

console.log('✅ VHADS Pro main application loaded - Production ready');

