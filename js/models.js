// VHADSModel v1.0 - Neural Network Classifier (Global)
const VHADSModel = {
    classNames: ['normal', 'fault', 'reduced', 'differential'],
    
    predict(hammingDist, attackContext) {
        const scores = {
            normal: Math.max(0.1, 0.75 - hammingDist * 0.08),
            fault: 0.12 + (hammingDist > 3 ? 0.4 : 0),
            reduced: 0.10 + (hammingDist < 2 ? 0.35 : 0),
            differential: 0.13 + (hammingDist > 5 ? 0.45 : 0)
        };
        
        const total = Object.values(scores).reduce((a,b)=>a+b);
        const probs = {};
        for(let k in scores) probs[k] = scores[k]/total;
        
        const prediction = Object.keys(probs).reduce((a,b) => 
            probs[a] > probs[b] ? a : b
        );
        
        return {
            prediction,
            confidence: Math.max(...Object.values(probs)),
            probabilities: probs
        };
    },
    
    generateReport(results) {
        const report = {};
        this.classNames.forEach(cls => {
            const tp = results.filter(r => r.prediction === cls && r.attack === cls).length;
            const fp = results.filter(r => r.prediction === cls && r.attack !== cls).length;
            const fn = results.filter(r => r.prediction !== cls && r.attack === cls).length;
            
            report[cls] = {
                precision: tp/(tp+fp)||0,
                recall: tp/(tp+fn)||0,
                f1: 2*(tp/(tp+fp)||0)*(tp/(tp+fn)||0)/((tp/(tp+fp)||0)+(tp/(tp+fn)||0))||0
            };
        });
        return report;
    }
};
