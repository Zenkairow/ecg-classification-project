# 🫀 CardioScan AI: Hierarchical ECG Diagnosis System

**CardioScan AI** is an advanced Multi-Stage Deep Learning system designed to diagnose cardiovascular pathologies from both raw signal data (`.npy`) and scanned ECG paper images.

## 🚀 Recent Updates: CardioScan Pro (v2.0)
We have just released the "Pro" production environment featuring:
- **Dual-Engine Core**:
  - **Engine A (Visual)**: Uses ResNet-50 for Computer Vision analysis of ECG images.
  - **Engine B (Signal)**: Uses a Hierarchical SE-ResNet-34 pipeline (Router -> Rhythm/Structure Specialists) for high-precision 12-lead signal analysis.
- **Professional UI**: A Streamlit-based interface with semantic health indicators and probability distributions.
- **Dockerized Deployment**: One-click launch via our `ecg-project` container.

---

## ⚠️ Important Note on Model Weights
**This repository DOES NOT contain the trained model weights (`.pth` files).**
Due to their large size and proprietary nature, the weights for the Router, Specialists, and Visual Engines are stored on our secure private server. 

**Access:** Authorized developers must download the models separately using the provided `setup_production.ps1` script (requires SSH credentials) or request access from the project maintainers.

---

## 🛠️ Quick Start (Production)

### Prerequisites
- Docker Desktop installed (with GPU support recommended).
- SSH Access to the Model Server (for initial setup).

### 1. Clone & Setup
Clone the repo and enter the directory:
```bash
git clone https://github.com/Zenkairow/ecg-classification-project.git
cd ecg-classification-project
```

### 2. Download Models (First Run Only)
Use our automation script to fetch the models and configure the production environment.
*(You will need the SSH password)*
```powershell
PowerShell -ExecutionPolicy Bypass -File .\setup_production.ps1
```

### 3. Launch the App
Run the Docker container script. This handles all dependencies and port forwarding.
```powershell
PowerShell -ExecutionPolicy Bypass -File .\run_docker_production.ps1
```

Access the dashboard at **[http://localhost:8501](http://localhost:8501)**.

---

## 📂 Project Structure
- `production/`: The self-contained deployment source code.
  - `app.py`: The Streamlit Frontend.
  - `utils/`: Inference backends for Visual and Signal engines.
- `src/`: Research and Training code.
  - `engine_a_visual/`: Training scripts for the Visual Classifier.
  - `engine_b_signal/`: Training scripts for the Signal Classifier.
- `models/`: Local storage for downloaded model weights (Ignored by Git).

---

## 🔬 Architecture Overview
1.  **Stage 1 (Gatekeeper)**: (Optional) Binary Normal vs Abnormal filter.
2.  **Stage 2 (The Router)**: Classifies the pathology type (Rhythm Issue vs Structural Issue).
3.  **Stage 3 (The Specialists)**:
    -   *Rhythm Net*: Fine-grained arrhythmia diagnosis (e.g., AFIB, SVT, Blocks).
    -   *Structure Net*: Morphology diagnosis (e.g., MI, Hypertrophy).

---
*© 2025 DeepMind Health / PBLS Team*
