# Brain Tumor Detection AI 🧠

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Launch%20App-success?style=for-the-badge&logo=github)](https://Arnav-102.github.io/NeuroScan/web/)

An AI-powered diagnostic tool capable of detecting brain tumors from MRI scans with high precision. This project uses a **ResNet18** deep learning model trained on over 10,000 images (via augmentation) and features a client-side web interface for instant analysis.

## 🌟 Features
- **High Accuracy**: Utilizes Transfer Learning with ResNet18.
- **Privacy-First**: Model runs entirely in the browser using ONNX Runtime (no data uploaded to server).
- **Interactive UI**: Professional, medical-grade web interface.

## 📂 Project Structure
```
brain_tumor_detection/
├── data/               # Place your 'Training' and 'Testing' folders here
├── model/
│   ├── train.py       # Script to train the ResNet18 model
│   └── export.py      # Script to export PyTorch model to ONNX
├── web/
│   ├── index.html     # Web App
│   ├── script.js      # Logic & Inference
│   └── style.css      # Styling
└── requirements.txt    # Dependencies
```

## 🚀 Getting Started

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
1. Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.
2. Extract the `Training` and `Testing` folders into the `data/` directory.
   - Example: `data/Training/glioma/...`

### 3. Train Model
Run the training script. This handles data augmentation (rotation, flip, zoom) to expand the effective dataset size.
```bash
python model/train.py
```
*This will save `brain_tumor_resnet18.pth`.*

### 4. Export for Web
Convert the trained PyTorch model to ONNX format for the web app.
```bash
python model/export.py
```
*This will create `web/model.onnx`.*

### 5. Launch App
Open `web/index.html` in your browser. You may need a local server if CORS issues arise:
```bash
cd web
python -m http.server 8000
```
Then visit `http://localhost:8000`.

## 📊 Performance
- **Target Metrics**: 85%+ Precision/Recall
- **Architecture**: ResNet18 (Pre-trained on ImageNet)
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary

---
*Created for portfolio demonstration.*
