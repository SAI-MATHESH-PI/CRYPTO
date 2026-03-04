// Charts v1.0 - Plotly Visualizations (Global)
const Charts = {
    init() {
        Plotly.newPlot('confusionMatrix', [{
            z: [[78,4,2,1],[3,72,5,2],[2,4,71,1],[1,2,3,70]],
            x:['Normal','Fault','Reduced','Diff'],
            y:['Normal','Fault','Reduced','Diff'],
            type:'heatmap',colorscale:'Blues',
            text: [[78,4,2,1],[3,72,5,2],[2,4,71,1],[1,2,3,70]],
            texttemplate:'%{text}'
        }],{title:'Confusion Matrix',width:450,height:350});
        
        Plotly.newPlot('trainingCurves', [{
            x:[1,5,10,15,20,25,30],y:[0.52,0.68,0.78,0.82,0.86,0.89,0.91],
            type:'scatter',mode:'lines+markers',name:'Train',
            line:{color:'#10b981'}
        },{
            x:[1,5,10,15,20,25,30],y:[0.49,0.65,0.74,0.79,0.82,0.85,0.87],
            type:'scatter',mode:'lines+markers',name:'Val',
            line:{color:'#1e40af',dash:'dash'}
        }],{title:'Training Curves',width:450,height:350});
        
        Plotly.newPlot('performanceMetrics', [{
            type:'scatterpolar',r:[0.92,0.89,0.905,0.898,0.94],
            theta:['Precision','Recall','F1','Accuracy','ROC'],
            fill:'toself',line:{color:'#1e40af'}
        }],{title:'Performance',width:450,height:350});
    },
    
    updateRealtime(data) {
        const colors = data.map(d=>d.prediction==='normal'?'#10b981':'#ef4444');
        Plotly.newPlot('realtimeChart', [{
            x:data.map(d=>d.frame),y:data.map(d=>d.hamming),
            mode:'lines+markers',type:'scatter',
            marker:{size:8,color:colors},
            line:{shape:'spline',width:3}
        }],{title:'Real-time Analysis',height:450});
    }
};
