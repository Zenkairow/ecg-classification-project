# CardioScan AI - Production Deployment

## Overview
This folder contains the self-contained production deployment for the Hierarchical ECG Classification System.

## Directory Structure
- `app.py`: The Main Streamlit Interface.
- `utils/`: 
    - `backend.py`: The inference engine (Coordinate Router & Specialists).
    - `model.py`: PyTorch model definitions.
- `models/`: Stores the trained `.pth` weights.

## Installation
1. Navigate to this folder:
   ```bash
   cd production
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App
Run the web interface locally:
```bash
# Ensure you are in the 'production' directory
streamlit run app.py
```
*Note: If running on a server with GPU, ensure `CUDA_VISIBLE_DEVICES` is set if needed.*
```bash
env CUDA_VISIBLE_DEVICES=1 streamlit run app.py
```

## Model Information
- **Stage 2 (Router)**: Distinguishes Rhythm vs Structure.
- **Stage 3 (Rhythm)**: Diagnoses specific arrhythmias (AFIB, SVT, etc.).
- **Stage 3 (Structure)**: Diagnoses morphology issues (MI, Hypertrophy, etc.).
