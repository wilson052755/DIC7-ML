// --- Configuration & State ---
let chart;
const state = {
    n: 500,
    noise: 100,
    seed: 42,
    data: [],
    model: { a: 0, b: 0, r2: 0, mse: 0 }
};

// --- DOM Elements ---
const nSlider = document.getElementById('n-slider');
const nVal = document.getElementById('n-val');
const noiseSlider = document.getElementById('noise-slider');
const noiseVal = document.getElementById('noise-val');
const seedInput = document.getElementById('seed-input');
const generateBtn = document.getElementById('generate-btn');
const predictX = document.getElementById('predict-x');
const predictY = document.getElementById('predict-y');

// --- Math Helpers ---
function seededRandom(seed) {
    const x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
}

function generateData() {
    state.data = [];
    let currentSeed = state.seed;
    
    // Ground truth random values
    const trueA = (seededRandom(currentSeed++) * 20) - 10;
    const trueB = (seededRandom(currentSeed++) * 100) - 50;
    const noiseMean = (seededRandom(currentSeed++) * 20) - 10;
    
    for (let i = 0; i < state.n; i++) {
        const x = (seededRandom(currentSeed++) * 200) - 100;
        // Simple normal-ish noise using Central Limit Theorem
        let noise = 0;
        for(let j=0; j<6; j++) noise += seededRandom(currentSeed++);
        noise = (noise / 6 - 0.5) * 2 * Math.sqrt(state.noise) * 2; // Scale noise
        
        const y = trueA * x + trueB + noise + noiseMean;
        state.data.push({ x, y });
    }
}

function trainModel() {
    const n = state.data.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0, sumYY = 0;
    
    for (const p of state.data) {
        sumX += p.x;
        sumY += p.y;
        sumXY += p.x * p.y;
        sumXX += p.x * p.x;
        sumYY += p.y * p.y;
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    // Calculate R2 and MSE
    let ssRes = 0;
    let ssTot = 0;
    const yMean = sumY / n;
    
    for (const p of state.data) {
        const yPred = slope * p.x + intercept;
        ssRes += Math.pow(p.y - yPred, 2);
        ssTot += Math.pow(p.y - yMean, 2);
    }
    
    state.model.a = slope;
    state.model.b = intercept;
    state.model.mse = ssRes / n;
    state.model.r2 = 1 - (ssRes / ssTot);
    
    updateUI();
}

function updateUI() {
    document.getElementById('mse-val').innerText = state.model.mse.toFixed(2);
    document.getElementById('r2-val').innerText = state.model.r2.toFixed(4);
    document.getElementById('learned-a').innerText = state.model.a.toFixed(2);
    
    updateChart();
    handlePredict();
}

function updateChart() {
    const scatterData = state.data.map(p => ({ x: p.x, y: p.y }));
    
    // Line data
    const minX = -100;
    const maxX = 100;
    const lineData = [
        { x: minX, y: state.model.a * minX + state.model.b },
        { x: maxX, y: state.model.a * maxX + state.model.b }
    ];

    if (chart) {
        chart.data.datasets[0].data = scatterData;
        chart.data.datasets[1].data = lineData;
        chart.update('none'); // Update without animation for performance
    } else {
        initChart(scatterData, lineData);
    }
}

function initChart(scatterData, lineData) {
    const ctx = document.getElementById('regressionChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: '資料點',
                    data: scatterData,
                    backgroundColor: 'rgba(56, 189, 248, 0.4)',
                    pointRadius: 3,
                },
                {
                    label: '回歸線',
                    data: lineData,
                    type: 'line',
                    borderColor: '#f43f5e',
                    borderWidth: 3,
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { 
                    type: 'linear', 
                    position: 'bottom',
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: { 
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f1f5f9' } }
            }
        }
    });
}

function handlePredict() {
    const x = parseFloat(predictX.value) || 0;
    const y = state.model.a * x + state.model.b;
    predictY.innerText = y.toFixed(4);
}

// --- Event Listeners ---
nSlider.addEventListener('input', (e) => {
    state.n = parseInt(e.target.value);
    nVal.innerText = state.n;
    runCycle();
});

noiseSlider.addEventListener('input', (e) => {
    state.noise = parseInt(e.target.value);
    noiseVal.innerText = state.noise;
    runCycle();
});

seedInput.addEventListener('change', (e) => {
    state.seed = parseInt(e.target.value);
    runCycle();
});

generateBtn.addEventListener('click', () => {
    state.seed = Math.floor(Math.random() * 1000);
    seedInput.value = state.seed;
    runCycle();
});

predictX.addEventListener('input', handlePredict);

function runCycle() {
    generateData();
    trainModel();
}

// Initial Run
runCycle();
