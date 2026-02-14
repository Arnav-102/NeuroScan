// NeuroScan Pro - System Controller V3.0

// [CONFIGURATION]
const CONFIG = {
    // Model Constants
    LABELS: ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
    MODEL_PATH: './model.onnx',
    THRESHOLD: 0.5,

    // UI Colors
    COLORS: {
        SAFE: '#10b981', // Emerald
        DANGER: '#ef4444', // Red
        WARNING: '#f59e0b', // Amber
        NEUTRAL: '#3b82f6' // Blue
    },

    // Medical Database
    INFO_DB: {
        'Glioma': "Abnormal growth of glial cells in brain/spine.\n\n[CAUSES]: Genetic mutations, exposure to radiation.\n[PRECAUTIONS]: Regular screening, avoiding radiation exposure where possible. Requires immediate neurosurgical consultation.",
        'Meningioma': "Tumor arising from the meninges.\n\n[CAUSES]: Hormonal imbalances, past radiation therapy, genetic disorders.\n[PRECAUTIONS]: Observation for small tumors, surgical resection for symptomatic ones. Regular MRI monitoring advised.",
        'Pituitary': "Abnormal growth in pituitary gland affecting hormones.\n\n[CAUSES]: Genetic conditions (MEN1), idiopathic factors.\n[PRECAUTIONS]: Hormone level monitoring, vision tests. Medication or surgery may be required to regulate hormone production.",
        'No Tumor': "No significant pathological anomalies detected.\n\n[ANALYSIS]: Brain structures appear normal.\n[PRECAUTIONS]: Maintain a healthy lifestyle, regular check-ups, and report any persistent headaches or neurological symptoms immediately."
    },


};

let session;

// [SYSTEM INITIALIZATION]
document.addEventListener('DOMContentLoaded', () => {
    initSystem();
    setupEventHandlers();
});



async function initSystem() {
    console.log("System Initializing...");
    try {
        if (typeof ort === 'undefined') throw new Error("ONNX Runtime Missing");

        session = await ort.InferenceSession.create(CONFIG.MODEL_PATH);
        console.log("Neural Engine Online.");

        // Update Status Dot to Green
        const statusDot = document.querySelector('.status-dot');
        if (statusDot) {
            statusDot.style.background = CONFIG.COLORS.SAFE;
            statusDot.style.boxShadow = `0 0 10px ${CONFIG.COLORS.SAFE}`;
        }

    } catch (e) {
        console.error("System Failure:", e);
        const statusDot = document.querySelector('.status-dot');
        if (statusDot) {
            statusDot.style.background = CONFIG.COLORS.DANGER;
            statusDot.innerText = " ERROR";
        }
        alert("CRITICAL ERROR: Failed to load Neural Engine.");
    }
}

// [EVENT HANDLERS]
function setupEventHandlers() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const resetBtn = document.getElementById('reset-btn');
    const downloadBtn = document.getElementById('download-btn');

    // Drag & Drop
    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = CONFIG.COLORS.NEUTRAL;
            dropZone.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '';
            dropZone.style.backgroundColor = '';
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length) handleInput(e.dataTransfer.files[0]);
        });

        dropZone.addEventListener('click', () => fileInput.click());
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) handleInput(e.target.files[0]);
        });
    }

    if (resetBtn) resetBtn.addEventListener('click', resetDashboard);
    if (downloadBtn) downloadBtn.addEventListener('click', generateReport);
}

// [CORE LOGIC]
function handleInput(file) {
    if (!file.type.startsWith('image/')) {
        alert("INVALID INPUT: Image sequence required.");
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        // UI Transition
        document.getElementById('drop-zone').classList.add('hidden');
        document.getElementById('preview-container').classList.remove('hidden');

        const img = document.getElementById('preview-image');
        img.src = e.target.result;
        img.style.opacity = '1'; // Ensure visible

        // Simulate Scanning Delay
        const wrapper = document.querySelector('.image-wrapper');
        wrapper.classList.add('scanning'); // Start Animation

        setTimeout(() => {
            document.getElementById('results-section').classList.remove('hidden');
            runAnalysis(img);
        }, 1500);
    };
    reader.readAsDataURL(file);
}

function resetDashboard() {
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('preview-container').classList.add('hidden');
    document.getElementById('drop-zone').classList.remove('hidden');
    document.getElementById('file-input').value = '';
    document.getElementById('preview-image').src = '';
    document.getElementById('preview-image').style.opacity = '0';

    // Reset Terminal
    document.getElementById('description-text').innerText = "AWAITING DATA...";

    // Reset Ring
    updateConfidenceRing(0, CONFIG.COLORS.NEUTRAL);

    // Reset Scanning Animation Elements
    const scannerBar = document.querySelector('.scanner-bar');
    const scanOverlay = document.querySelector('.scan-overlay-grid');
    const wrapper = document.querySelector('.image-wrapper');

    if (wrapper) wrapper.classList.remove('scanning');
    if (scannerBar) scannerBar.style.display = 'block';
    if (scanOverlay) scanOverlay.style.display = 'block';
}

// [INFERENCE ENGINE]
async function runAnalysis(imageElement) {
    if (!session) return;

    try {
        const tensor = await preprocessImage(imageElement);
        const feeds = { input: tensor };
        const results = await session.run(feeds);
        const output = results.output.data;
        const probabilities = softmax(Array.from(output));

        presentResults(probabilities);

    } catch (e) {
        console.error("Inference Error:", e);
    }
}

// [PREPROCESSING]
async function preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, 224, 224);
    const { data } = ctx.getImageData(0, 0, 224, 224);

    const float32Data = new Float32Array(3 * 224 * 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < 224 * 224; i++) {
        float32Data[i] = ((data[i * 4] / 255) - mean[0]) / std[0]; // R
        float32Data[i + 224 * 224] = ((data[i * 4 + 1] / 255) - mean[1]) / std[1]; // G
        float32Data[i + 2 * 224 * 224] = ((data[i * 4 + 2] / 255) - mean[2]) / std[2]; // B
    }
    return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

// [UI PRESENTATION]
function presentResults(probabilities) {
    // Sort Results
    const sorted = probabilities
        .map((p, i) => ({ label: CONFIG.LABELS[i], prob: p }))
        .sort((a, b) => b.prob - a.prob);

    const topResult = sorted[0];
    const topLabel = topResult.label;
    const confidence = topResult.prob * 100;

    // 1. Update Primary Card
    const labelEl = document.getElementById('top-prediction-label');
    labelEl.innerText = topLabel.toUpperCase();

    // Dynamic Colouring
    let statusHVal = CONFIG.COLORS.NEUTRAL;
    if (topLabel === 'No Tumor') statusHVal = CONFIG.COLORS.SAFE;
    else if (confidence > 80) statusHVal = CONFIG.COLORS.DANGER;
    else statusHVal = CONFIG.COLORS.WARNING;

    // Apply Colors
    document.documentElement.style.setProperty('--accent-color', statusHVal);
    document.querySelector('.primary-result-card').style.borderColor = statusHVal;

    // 2. Animate Ring
    updateConfidenceRing(confidence, statusHVal);

    // 3. Populate Data Grid
    const container = document.getElementById('predictions-container');
    container.innerHTML = '';
    sorted.forEach(item => {
        const pct = (item.prob * 100).toFixed(1);
        const div = document.createElement('div');
        div.className = 'data-card';
        div.innerHTML = `
            <div class="label">${item.label}</div>
            <div style="display:flex; justify-content:space-between; font-size:0.9rem;">
                <span>PROBABILITY</span>
                <span>${pct}%</span>
            </div>
            <div class="bar">
                <div class="bar-fill" style="width: ${pct}%; background: ${item.label === topLabel ? statusHVal : '#555'}"></div>
            </div>
        `;
        container.appendChild(div);
    });

    // 4. Typewriter Terminal Effect
    const descText = CONFIG.INFO_DB[topLabel] || "Analysis inconclusive.";
    typeWriterEffect(document.getElementById('description-text'), `>> DETECTED: ${topLabel}\n>> CONFIDENCE: ${confidence.toFixed(2)}%\n>> NOTES: ${descText}`);

    // 5. Stop Scanning Animation
    const scannerBar = document.querySelector('.scanner-bar');
    const scanOverlay = document.querySelector('.scan-overlay-grid');
    const wrapper = document.querySelector('.image-wrapper');

    if (wrapper) wrapper.classList.remove('scanning');
    if (scannerBar) scannerBar.style.display = 'none';
    if (scanOverlay) scanOverlay.style.display = 'none';
}

function updateConfidenceRing(percent, color) {
    const circle = document.querySelector('.progress-ring__circle');
    const radius = circle.r.baseVal.value;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (percent / 100) * circumference;

    circle.style.strokeDashoffset = offset;
    circle.style.stroke = color;

    // Count up text
    const valueEl = document.querySelector('.confidence-value');
    let current = 0;
    const timer = setInterval(() => {
        current += 1;
        valueEl.innerText = current + "%";
        if (current >= Math.floor(percent)) {
            clearInterval(timer);
            valueEl.innerText = percent.toFixed(1) + "%";
        }
    }, 10);
}

function typeWriterEffect(element, text) {
    element.innerText = "";
    let i = 0;
    const speed = 20;

    function type() {
        if (i < text.length) {
            element.innerText += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    type();
}

// [REPORT GENERATION]
function generateReport() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    // Data Retrieval
    const timestamp = new Date().toLocaleString();
    const uniqueId = "NS-" + Date.now().toString().slice(-6);
    const topLabel = document.getElementById('top-prediction-label').innerText;
    const confidence = document.querySelector('.confidence-value').innerText;
    const imgData = document.getElementById('preview-image').src;

    // DB Lookup
    const dbKey = Object.keys(CONFIG.INFO_DB).find(key => key.toUpperCase() === topLabel) || "No Tumor";
    const description = CONFIG.INFO_DB[dbKey];

    // [MINIMALIST STYLING]
    const primaryColor = [20, 20, 20]; // Almost Black
    const secondaryColor = [100, 100, 100]; // Gray

    // 1. Header (Clean & Simple)
    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.setTextColor(...primaryColor);
    doc.text("NeuroScan AI", 20, 20);

    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(...secondaryColor);
    doc.text("Diagnostic Analysis Report", 20, 26);

    // Meta Data (Right Side)
    doc.text(`ID: ${uniqueId}`, 190, 20, { align: "right" });
    doc.text(`DATE: ${timestamp}`, 190, 26, { align: "right" });

    doc.setDrawColor(200);
    doc.setLineWidth(0.5);
    doc.line(20, 35, 190, 35);

    // 2. Results Section (Top Left)
    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.setTextColor(...primaryColor);
    doc.text("FINDINGS", 20, 50);

    doc.setFontSize(12);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(...secondaryColor);
    doc.text("Classification:", 20, 60);
    doc.text("Confidence:", 20, 70);

    doc.setFont("helvetica", "bold");
    doc.setTextColor(...primaryColor);
    doc.text(topLabel, 60, 60);
    doc.text(confidence, 60, 70);

    // 3. Patient Scan Image (Small & Right Aligned)
    try {
        // Image Size: 40x40 (Much smaller)
        const imgSize = 40;
        const imgX = 150;
        const imgY = 45;

        // Thin border
        doc.setDrawColor(220);
        doc.rect(imgX, imgY, imgSize, imgSize);
        doc.addImage(imgData, 'JPEG', imgX, imgY, imgSize, imgSize);

        doc.setFontSize(8);
        doc.setTextColor(...secondaryColor);
        doc.text("Scan Source", imgX + (imgSize / 2), imgY + imgSize + 5, { align: "center" });
    } catch (e) {
        console.error("Image Error", e);
    }

    // 4. Medical Context (Clean Text)
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(...primaryColor);
    doc.text("MEDICAL CONTEXT", 20, 110);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(...secondaryColor);
    const splitDesc = doc.splitTextToSize(description, 170);
    doc.text(splitDesc, 20, 120);

    // 5. Footer (Minimal)
    doc.setDrawColor(200);
    doc.line(20, 270, 190, 270);

    doc.setFontSize(8);
    doc.setTextColor(150);
    doc.text("NeuroScan AI System V3.0 | Automated Analysis", 20, 278);
    doc.text("Verification required by qualified professional.", 190, 278, { align: "right" });

    doc.save(`NeuroScan_${uniqueId}.pdf`);
}
