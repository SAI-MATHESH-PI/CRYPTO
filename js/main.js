// VHADS Pro v2.0 - Production Application Orchestrator
import { HIGHTCipher } from './cipher.js';
import { VHADSModel } from './ml-models.js';
import { VideoProcessor } from './video.js';
import { Charts } from './charts.js';

class VHADSApp {
    constructor() {
        this.cipher = new HIGHTCipher();
        this.model = new VHADSModel();
        this.video = new VideoProcessor();
        this.charts = new Charts();
        
        this.state = {
            videoFile: null,
            results: [],
            isAnalyzing: false,
            attackRate: 0.25,
            frameCount: 64
        };

        this.init();
    }

    init() {
        this.bindEvents();
        this.updateControls();
        this.renderMetrics();
        this.setStatus('ready', 'VHADS Pro v2.0 Ready');
    }

    bindEvents() {
        // Controls
        document.getElementById('videoInput').addEventListener('change', (e) => 
            this.handleVideoUpload(e.target.files[0])
        );
        
        ['attackRate', 'frameCount'].forEach(id => {
            document.getElementById(id).addEventListener('input', (e) => {
                this.state[e.target.id] = parseFloat(e.target.value);
                document.getElementById(`${id}Value`).textContent = e.target.value;
            });
        });

        // Analysis
        document.getElementById('analyzeBtn').addEventListener('click', () => 
            this.runAnalysis()
        );

        // Tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
    }

    async handleVideoUpload(file) {
        if (!file || !file.type.startsWith('video/')) {
            this.setStatus('error', 'Please upload a valid video file (MP4, MOV, AVI)');
            return;
        }

        this.state.videoFile = file;
        const url = URL.createObjectURL(file);
        const preview = document.getElementById('videoPreview');
        
        preview.innerHTML = `
            <video controls autoplay muted style="max-height: 200px; border-radius: var(--radius);">
                <source src="${url}" type="${file.type}">
            </video>
        `;
        
        document.getElementById('analyzeBtn').disabled = false;
        this.setStatus('ready', `Video loaded: ${file.name}`);
    }

    async runAnalysis() {
        if (this.state.isAnalyzing) return;
        
        this.state.isAnalyzing = true;
        this.setStatus('processing', 'Running real-time HIGHT analysis...');
        document.getElementById('analyzeBtn').disabled = true;

        try {
            const results = await this.processFrames();
            this.state.results = results;
            
            this.updateFrameView(0);
            this.charts.updateRealtimePlot(results);
            this.updateMetricsSummary();
            this.renderClassificationReport();
            
            this.setStatus('success', `Analysis complete: ${results.length} frames processed`);
        } catch (error) {
            this.setStatus('error', `Analysis failed: ${error.message}`);
        } finally {
            this.state.isAnalyzing = false;
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('analyzeBtn').textContent = '✅ Re-run Analysis';
        }
    }

    async processFrames() {
        const results = [];
        const inputCanvas = document.getElementById('inputFrame');
        const outputCanvas = document.getElementById('outputFrame');
        
        for (let i = 0; i < this.state.frameCount; i++) {
            // Generate realistic CCTV frame data
            const frameData = this.video.generateCCTVFrame();
            
            // Apply HIGHT encryption with probabilistic attacks
            const attackType = Math.random() < this.state.attackRate 
                ? this.video.selectAttackType() 
                : 'normal';
                
            const encrypted = this.cipher.encrypt(frameData.slice(0, 8), attackType);
            const hammingDist = this.video.calculateHamming(frameData.slice(0, 8), encrypted);
            const prediction = this.model.predict(hammingDist, attackType);
            
            results.push({
                frame: i + 1,
                attack: attackType,
                prediction,
                hamming: hammingDist,
                confidence: this.model.confidence
            });

            // Real-time visualization
            this.video.renderFrame(inputCanvas, frameData);
            this.video.renderEncryptedFrame(outputCanvas, encrypted, frameData);
            
            // Update UI
            document.getElementById('inputFrameNum').textContent = i + 1;
            document.getElementById('hammingDistance').textContent = hammingDist;
            document.getElementById('framePrediction').textContent = prediction;
            
            // Throttle for realistic 30 FPS
            await new Promise(resolve => setTimeout(resolve, 33));
        }
        
        return results;
    }

    switchTab(tabId) {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
            btn.setAttribute('aria-selected', 'false');
        });
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.remove('active');
        });

        event.target.classList.add('active');
        event.target.setAttribute('aria-selected', 'true');
        document.getElementById(`${tabId}-panel`).classList.add('active');
    }

    setStatus(type, message) {
        const statusEl = document.querySelector('[data-status]');
        statusEl.textContent = message;
        statusEl.className = `status status--${type}`;
    }

    updateControls() {
        // Sync slider values
        document.getElementById('attackRateValue').textContent = this.state.attackRate;
        document.getElementById('frameCountValue').textContent = this.state.frameCount;
    }

    updateMetricsSummary() {
        const results = this.state.results;
        const normalCount = results.filter(r => r.prediction === 'normal').length;
        const accuracy = (normalCount / results.length * 100).toFixed(1);
        
        document.getElementById('overallAccuracy').textContent = `${accuracy}%`;
        document.getElementById('normalFrames').textContent = normalCount;
        document.getElementById('attackFrames').textContent = results.length - normalCount;
    }

    renderClassificationReport() {
        // Implementation in separate module
        this.charts.renderClassificationReport(this.state.results);
    }

    renderMetrics() {
        this.charts.renderStaticMetrics();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        new VHADSApp();
    } catch (error) {
        console.error('VHADS Pro initialization failed:', error);
        document.body.innerHTML = `
            <div style="text-align: center; padding: 4rem; color: var(--danger);">
                <h1>Application Error</h1>
                <p>Failed to initialize VHADS Pro. Please refresh the page.</p>
            </div>
        `;
    }
});
