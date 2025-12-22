# 🫀 CardioScan AI: Technical Architecture & Feature Engineering Report
**Version 2.0 | December 2025**

This document provides an exhaustive technical breakdown of the CardioScan AI system, detailing the feature engineering, architectural choices, and training strategies for every model file in the repository.

---

## 1. 🟢 Production Models (Active Deployment)
These models are currently live in the `production/` environment.

### 1.1 `hierarchy_stage2_router.pth` (The Gatekeeper)
*   **Role**: **Engine B - Signal Router**
*   **Objective**: Binary Classification -> **Rhythm** (Electrical) vs **Structure** (Physical).
*   **Architecture**: `SEResNet34` (Squeeze-and-Excitation ResNet).
    *   *Why?* The SE-Blocks provide channel-wise attention, effectively allowing the model to "focus" on specific ECG leads (e.g., Lead II for Rhythm) while ignoring others.
*   **Hyperparameters & Strategy**:
    *   **Loss Function**: **Focal Loss** ($\gamma=2.0$). Down-weights easy examples (Normal sinus rhythm) to focus learning on hard/rare classes.
    *   **Sampler**: `WeightedRandomSampler`. Forces training batches to be perfectly balanced (50/50) despite the dataset being 90% Normal.
    *   **Learning Rate**: `3e-4` (AdamW Optmizer, Weight Decay `1e-2`).
    *   **Input**: 12-Lead Signal (500Hz, 5000 samples). Interpolated if length mismatch.

### 1.2 `hierarchy_stage3_rhythm.pth` (The Arrhythmia Specialist)
*   **Role**: **Engine B - Rhythm Specialist**
*   **Objective**: Diagnoses specific electrical faults (AFIB, SVT, AV Blocks).
*   **Architecture**: `SEResNet34`.
*   **Hyperparameters & Strategy**:
    *   **Optimization**: Trained with **CrossEntropyLoss** and `WeightedRandomSampler` to handle intra-class imbalance (e.g., rarely seen AVB vs common AFIB).
    *   **Learning Rate**: `1e-4` with cosine annealing.
    *   **Input**: Same 12-lead signal, but the model learns to prioritize **Time-Domain features** (R-R intervals, P-wave absence).

### 1.3 `hierarchy_stage3_structure.pth` (The Morphology Specialist)
*   **Role**: **Engine B - Structure Specialist**
*   **Objective**: Diagnoses physical damage (Myocardial Infarction, Hypertrophy).
*   **Architecture**: `SEResNet34`.
*   **Feature Engineering (Crucial)**:
    *   **Gaussian Noise Injection**: During training, random 1% Gaussian noise is added to the signal (`signal + noise`).
    *   *Why?* Structural diagnoses rely on **Morphology** (ST-elevation shapes). Noise forces the model to learn the *global shape* of the wave rather than overfitting to high-frequency artifacts or specific lead noise.
    *   **Scheduler**: `ReduceLROnPlateau` (Patience=3). Aggressively drops LR when validation loss stalls to find the deepest minimum.

### 1.4 `Engine_A_ResNet-50.pth` (The Visual Eye)
*   **Role**: **Engine A - Visual Classifier**
*   **Objective**: Diagnosis from 2D Paper ECG Images.
*   **Architecture**: **ResNet-50** (ImageNet Pretrained).
*   **Feature Engineering**:
    *   **Synthetic Data Pipeline**: Trained on images generated with:
        *   Artificial Red/Green Grid Backgrounds.
        *   Random Perspective Warping (simulating phone camera angles).
        *   Gaussian Blur & Shadow Injection.
    *   *Outcome*: The model treats the grid as "transparent noise" and locks onto the signal trace.

---

## 2. 🟡 Research & Experimental Models
Found in `models/`, these represent alternative approaches or backup systems.

### 2.1 `classifier_efficientnet_b4.pth`
*   **Architecture**: **EfficientNet-B4**.
*   **Strategy**: Trained on High-Resolution (1024x1024) images.
*   **Augmentation**: Uses `RandomAffine` (5° rotation, scaling), `ColorJitter`, and `RandomErasing` (Cutout).
*   **Status**: Higher parameter efficiency than ResNet, but slightly slower inference. Kept as a high-accuracy backup.

### 2.2 `hydra_fusion_best.pth` (The HydraNet)
*   **Architecture**: **Multi-Branch CNN**.
*   **Ideation**: Inspired by clinical anatomy. It splits the 12 leads into 3 groups:
    1.  **Anterior Branch** (V1-V4)
    2.  **Lateral Branch** (I, aVL, V5, V6)
    3.  **Inferior Branch** (II, III, aVF)
*   **Tech**: Each branch processes its leads independently, and their outputs are concatenated ("Fused") into a dense layer at the end.
*   **Status**: Experimental. Theoretically superior for localizing infarcts but computationally expensive.

### 2.3 `stage1_gatekeeper.pth`
*   **Objective**: **Binary Normal/Abnormal Filter**.
*   **Strategy**: Tuned for **High Recall (>99%)**.
*   **Loss**: `BCEWithLogitsLoss` with `pos_weight` set to penalize missing a sick patient 10x more than flagging a healthy one.

---

## 3. 🟠 Legacy Development Artifacts
Models produced during the iterative R&D phase.

*   `signal_resnet50_69acc.pth`: Attempt at using a deeper ResNet-50 for signals. **Failed (Overfitting)**. Proved that for 1D signals, deeper isn't always better; ResNet-34 is the "sweet spot."
*   `signal_transformer_baseline.pth`: Pure Transformer architecture. **Failed (Data Starvation)**. Transformers need millions of samples; we have thousands.
*   `ecg_model_v[2-5].pth`: Early 1D-CNN prototypes. Lacked **Residual Connections** (skip connections), causing the "Vanishing Gradient" problem where accuracy capped at 60%.

---

## 4. 🛠️ Utilities

### `RealESRGAN_x4plus.pth`
*   **Type**: GAN (Generative Adversarial Network).
*   **Function**: **Super-Resolution**. Upscales pixelated, low-quality smartphone photos of ECGs by 4x before feeding them to Engine A.

---

**Glossary of Techniques:**
*   **SE-Block**: A module that re-calibrates channel weights (Attention).
*   **Focal Loss**: A loss function that reduces the weight of easy examples.
*   **Mixup**: Blending two images/signals to create a "ghost" sample.
*   **WeightedRandomSampler**: artificially re-balancing a dataset by sampling rare items more often.

*Report Generated by Antigravity (Google DeepMind) for CardioScan AI Project.*
