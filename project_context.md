# PROJECT CONTEXT — AI-Powered Cardiac Diagnostic Platform
Migration Package for Antigravity (Full Control Mode)

## 1. Project Overview
We are building an AI-powered cardiac diagnostic system to digitize and analyze 12-lead ECG printouts.

### System Pipeline
1. Input: Mobile photo of ECG printout  
2. Model 1 (Restorer): GAN-based enhancement using Real-ESRGAN  
3. Model 2 (Extractor): CNN + Transformer model → numerical 12-lead signals  
4. Model 3 (Classifier): Transformer-based arrhythmia + demographic prediction  
5. Output: Digital ECG + diagnosis results  

### Purpose
- Convert analog ECGs to digital
- Enable telemedicine & remote diagnosis
- Build ECG datasets for research

### Technologies
- Python, PyTorch
- Real-ESRGAN (basicsr)
- Albumentations
- OpenCV
- Transformers

### Infrastructure
- VIIT HPC Server (NVIDIA A100 MIG ~10GB)
- Controlled dependency environment

---

## 2. Completed Work

### A. Model 1 — Restorer (GAN)
- Architecture: RRDBNet via Real-ESRGAN/basicsr  
- Task-Aligned Training (Damaged → Signal-Only target)
- Trained for 50 epochs  
- Output verifies successful training: `y_fake_epoch_49.png`

### B. Data Engineering
- 21,799 “signal-only” ground-truth masks generated using HSV extraction  
- Heavy augmentation pipeline implemented in `train_gan.py`

### C. Environment
- Requirements pinned:  
  - torch==2.1.0  
  - torchvision==0.16.0  
  - numpy==1.26.4  
- Virtualenv + nohup pipeline ready

### D. Documentation
- Full 8-chapter design report
- DFD level 0/1
- ER diagram
- UML Class, State, Sequence diagrams

---

## 3. Remaining Work (Antigravity Tasks)

### A. Model 1 Validation
- Implement MSE  
- Implement DTW  
- Perform fidelity comparison

### B. Model 2 — Extractor (CNN + Transformer)
- Finalize architecture  
- Create training scripts  
- Define output format: 12-lead `.npy` arrays  
- Implement dataloader for restored images

### C. Validation Gate
- Detect flatlines  
- Detect noise  
- Define signal integrity thresholds  

### D. Model 3 — v6 Classifier
Two training stages:
1. SSL pretraining  
2. Multi-task fine-tuning (Arrhythmia, Age, Sex)

Needed:
- Architecture  
- Loss functions  
- Metrics  
- Training pipeline

### E. Application Layer
- Implement AppController  
- Database models: User, Scan, Diagnosis  
- API integration (FastAPI recommended)  
- Connect all models to unified pipeline

---

## 4. Agent Alignment Requirements
- Antigravity must NOT modify files unless approved.  
- All changes must be PROPOSED, never executed.  
- Use diffs for every modification.  
- Reason → Debate → Propose → WAIT for approval.

This document is the single source of truth for the entire project.
