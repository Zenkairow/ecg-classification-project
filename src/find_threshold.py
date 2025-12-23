import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import recall_score, confusion_matrix

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

# Imports
from src.engine_b_signal.train_signal_model import ECGSignalDataset, CSV_PATH, DATA_DIR, NUM_LEADS
from src.engine_b_signal.models.resnet1d_se import SEResNet34

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

def find_optimal_threshold():
    print("--- Calibrating Stage 1 Threshold for 99% Recall ---")
    
    # 1. Data
    print("Loading Validation Set...")
    dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, detected_seq_len=5000)
    
    # Normal Indices
    normal_indices = set()
    for name, idx in dataset.class_map.items():
        if name in ['NORM', 'Sinus_Rhythm']:
            normal_indices.add(idx)
            
    # Split (Same seed as training)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model
    model = SEResNet34(num_classes=1, input_channels=NUM_LEADS) # Binary Logit
    model = model.to(DEVICE)
    
    # Load Model
    # Try expected path, fallback to user mentioned path if needed, though I wrote it as stage1_gatekeeper.pth
    path = "models/stage1_gatekeeper.pth"
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Checking alternate names...")
        if os.path.exists("models/hierarchy_stage1_gatekeeper.pth"):
             path = "models/hierarchy_stage1_gatekeeper.pth"
        else:
             print("Error: Could not find model file.")
             return
             
    print(f"Loading weights from {path}")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    
    # 3. Inference
    print("Running Inference...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, original_labels in tqdm(val_loader):
            inputs = inputs.to(DEVICE)
            
            # Binary Labels
            binary_labels = [0 if l.item() in normal_indices else 1 for l in original_labels]
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(binary_labels)
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 4. Threshold Search
    print("\n--- Sweeping Thresholds (0.01 - 0.50) ---")
    print(f"{'Threshold':<10} | {'Recall (Abnormal)':<20} | {'Specificity (Normal)':<20} | {'Status'}")
    print("-" * 65)
    
    best_threshold = 0.01
    best_spec = 0.0
    found_target = False
    
    # Scan from high to low (or low to high). 
    # We want HIGHEST threshold that keeps Recall >= 0.99 (because higher threshold = better specificity)
    thresholds = np.arange(0.01, 0.51, 0.01)
    
    results = []
    
    for thr in thresholds:
        preds = (all_probs >= thr).astype(int)
        
        # Recall = TP / (TP + FN)  (Abnormal Acc)
        # Specificity = TN / (TN + FP) (Normal Acc)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append((thr, recall, specificity))
        
        status = ""
        if recall >= 0.99:
            status = "PASS"
            if thr > best_threshold:
                best_threshold = thr
                best_spec = specificity
                found_target = True
        else:
            status = "FAIL (<99%)"
            
        # Print valid candidates or steps
        if int(thr * 100) % 5 == 0: # Print every 0.05
             print(f"{thr:<10.2f} | {recall:<20.2%} | {specificity:<20.2%} | {status}")
             
    print("-" * 65)
    
    if found_target:
        print(f"\n✅ OPTIMAL THRESHOLD FOUND: {best_threshold:.2f}")
        print(f"Recall: >= 99.0%")
        print(f"Specificity: {best_spec:.2%}")
        print(f"Action: Set THRESHOLD = {best_threshold:.2f} in service.py")
    else:
        print("\n❌ FAILED. No threshold achieved 99% Recall.")
        # Find max recall
        max_rec = max([r[1] for r in results])
        print(f"Max Recall Possible: {max_rec:.2%}")

if __name__ == "__main__":
    find_optimal_threshold()
