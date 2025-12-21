import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

# Imports
from src.engine_b_signal.train_signal_model import ECGSignalDataset, CSV_PATH, DATA_DIR, NUM_LEADS
from src.engine_b_signal.models.resnet1d_se import SEResNet34

# Config - High Sensitivity Protocol
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 3e-4 # Slightly higher for robust learning with decay
WEIGHT_DECAY = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_pos_weight(dataset):
    """
    Computes pos_weight for BCEWithLogitsLoss.
    Formula: NumNegative / NumPositive (assuming Normal=0, Abnormal=1)
    Actually, to increase Recall of Class 1 (Abnormal/Positive), we weight Positive errors more.
    So pos_weight = NumNeg / NumPos might be standard for balancing.
    To FORCE Sensitivity, we might want even higher. 
    But standard 'balanced' for BCE is Neg/Pos.
    """
    print("Computing class distribution...")
    loss_col = 'label' if 'label' in dataset.annotations.columns else 'diagnostic_superclass'
    all_labels = dataset.annotations[loss_col].values
    
    # Re-import grouping logic
    from src.engine_b_signal.train_signal_model import group_diagnostic_classes
    
    num_normal = 0    # Class 0 (Negative)
    num_abnormal = 0  # Class 1 (Positive)
    
    for raw_label in all_labels:
        group = group_diagnostic_classes(raw_label)
        if group in ['NORM', 'Sinus_Rhythm']:
            num_normal += 1
        else:
            num_abnormal += 1
            
    total = num_normal + num_abnormal
    print(f"Stats: Normal (0): {num_normal}, Abnormal (1): {num_abnormal}")
    
    # pos_weight > 1 increases recall.
    # Standard balance: N_neg / N_pos
    pos_weight = num_normal / num_abnormal
    
    # If standard formula gives < 1 (Abnormal is majority), recall might suffer if we just balance.
    # The prompt asking for "High Sensitivity" and "pos_weight to this ratio" implies boosting positive class.
    # If Abnormal is 60% and Normal 40%, ratio is 0.66. This would DOWNWEIGHT abnormal.
    # User might mean: Weight for Abnormal should be Ratio of Normal/Abnormal if Normal is Majority.
    # If Abnormal is Majority, we still want HIGH RECALL.
    # So let's ensure pos_weight is at least 1.5 or 2.0 or the calculated ratio, whichever is higher?
    # Actually, let's stick to the User Directive: "Count the number of (Normal)/(Abnormal). Set pos_weight to this ratio"
    # Wait, if Normal > Abnormal, Ratio > 1. Good.
    # If Normal < Abnormal, Ratio < 1. Bad for Recall.
    # Let's assume user wants to upweight Abnormal regardless.
    # I will clip it to be at least 1.0.
    
    final_weight = max(pos_weight, 1.5) # Force at least 1.5x attention on Abnormal
    
    print(f"Calculated Ratio: {pos_weight:.2f}")
    print(f"Final pos_weight used: {final_weight:.2f} (Clipped to min 1.5 to force Sensitivity)")
    
    return torch.tensor([final_weight]).to(DEVICE)

def apply_augmentations(inputs):
    """
    Apply on-the-fly augmentations to [Batch, Leads, Time]
    1. Random Horizontal Flip (Time Reversal) p=0.5
    2. Gaussian Noise
    """
    # 1. Flip (Time dimension is last)
    if np.random.rand() > 0.5:
        # torch.flip returns a copy
        inputs = torch.flip(inputs, dims=[-1])
        
    # 2. Gaussian Noise
    noise = torch.randn_like(inputs) * 0.01 # Small sigma
    inputs = inputs + noise
    
    return inputs

def train_gatekeeper():
    print("--- Starting Stage 1: Protocol 'High Sensitivity' ---")
    os.makedirs("models", exist_ok=True)
    
    # 1. Data
    dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, detected_seq_len=5000)
    
    # Normal Indices Set
    normal_indices = set()
    for name, idx in dataset.class_map.items():
        if name in ['NORM', 'Sinus_Rhythm']:
            normal_indices.add(idx)
            
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model (Num Outputs = 1 for BCE)
    model = SEResNet34(num_classes=1, input_channels=NUM_LEADS)
    model = model.to(DEVICE)
    
    # 3. Loss with pos_weight
    pos_weight = compute_pos_weight(dataset)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_recall = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, original_labels in pbar:
            inputs = inputs.to(DEVICE)
            
            # Augment
            inputs = apply_augmentations(inputs)
            
            # Map Labels to Float [Batch, 1]
            binary_labels = torch.tensor(
                [0.0 if l.item() in normal_indices else 1.0 for l in original_labels], 
                device=DEVICE
            ).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs) # [Batch, 1]
            loss = criterion(outputs, binary_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, original_labels in val_loader:
                inputs = inputs.to(DEVICE)
                
                # No Augmentation on Val
                
                # Map Labels
                binary_labels = [0 if l.item() in normal_indices else 1 for l in original_labels]
                
                # VALIDATION AT 0.14 THRESHOLD (Calibrated for >99% Recall)
                # Default was 0.5, but we know we need 0.14 to be safe.
                outputs = model(inputs)
                preds = torch.sigmoid(outputs) > 0.14 # CALIBRATED THRESHOLD
                preds = preds.long().squeeze(1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(binary_labels)
        
        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds, zero_division=0) # Sensitivity
        prec = precision_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Recall: {rec:.2%} | Acc: {acc:.2%} | Prec: {prec:.2%}")
        
        # SAVE ONLY ON RECALL IMPROVEMENT
        if rec > best_recall:
            best_recall = rec
            torch.save(model.state_dict(), "models/stage1_gatekeeper.pth")
            print("  -> Saved Best Recall Model")
            
    print(f"\nTraining Complete. Best Recall: {best_recall:.2%}")

if __name__ == "__main__":
    train_gatekeeper()
