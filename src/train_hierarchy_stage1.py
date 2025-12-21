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

# Config
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_binary_label(original_label_idx, class_map):
    """
    Remaps 25 classes to 2 classes:
    0: Normal (NORM, Sinus_Rhythm)
    1: Abnormal (Everything else)
    """
    # Invert map to get string name
    idx_to_class = {v: k for k, v in class_map.items()}
    label_name = idx_to_class[original_label_idx]
    
    if label_name in ['NORM', 'Sinus_Rhythm']:
        return 0
    else:
        return 1

def compute_class_weights(dataset, normal_indices):
    """
    Computes weights dynamically.
    Strategy:
    1. Count Normal vs Abnormal.
    2. Compute Inverse Frequency (Balanced Weights).
    3. FORCE Abnormal Weight to be at least Equal or Higher to Normal to guarantee Recall.
    """
    print("Computing class weights from dataset...")
    
    # Efficient counting
    # Access the label column directly
    loss_col = 'label' if 'label' in dataset.annotations.columns else 'diagnostic_superclass'
    all_labels = dataset.annotations[loss_col].values
    
    # We need to map these to indices to check against normal_indices
    # dataset.class_map maps string -> int
    # But dataset.annotations has strings?
    # Let's check a sample.
    example = all_labels[0]
    
    # Map all to binary
    # We can use the class_map to get the int index, then check normal_indices
    # BUT dataset.annotations might have raw strings that need grouping first?
    # dataset.unique_labels has the grouped labels.
    # dataset.annotations has raw labels.
    # We need to use `dataset.class_map` logic.
    
    num_normal = 0
    num_abnormal = 0
    
    # Invert map for fast check if we had indices, but we have strings
    # Grouping logic is: group_diagnostic_classes(raw) -> group
    # We don't have that function imported easily without circular import if we aren't careful.
    # Actually we imported `group_diagnostic_classes`? No, we didn't.
    # Let's import it or re-implement simple check.
    
    # Easier: Iterate the dataset indices (which returns loaded items) is slow.
    # Fast way:
    inverted_map = {v: k for k, v in dataset.class_map.items()}
    
    # Iterate all strings, group them, check against Normal
    # Wait, dataset.__getitem__ does the grouping.
    # Let's trust the distribution given by user: 40% Normal.
    # If we want to be exact, we can loop. Since 21k is small, let's loop the dataframe.
    
    # Re-import grouping logic
    from src.engine_b_signal.train_signal_model import group_diagnostic_classes
    
    for raw_label in all_labels:
        group = group_diagnostic_classes(raw_label)
        if group in ['NORM', 'Sinus_Rhythm']:
            num_normal += 1
        else:
            num_abnormal += 1
            
    total = num_normal + num_abnormal
    print(f"Distribution: Normal={num_normal} ({num_normal/total:.1%}), Abnormal={num_abnormal} ({num_abnormal/total:.1%})")
    
    # Compute Balanced Weights: Total / (NumClasses * ClassCount)
    w_norm = total / (2 * num_normal)
    w_abnorm = total / (2 * num_abnormal)
    
    print(f"Theoretical Balanced Weights: Normal={w_norm:.2f}, Abnormal={w_abnorm:.2f}")
    
    # USER CONSTRAINT: "Maximize Recall of Abnormal"
    # This means we punish Missing Abnormal (False Negative).
    # So w_abnorm should be High.
    # USER CONSTRAINT: "Weight Normal Slightly Higher" (Potential Confusion)
    # If Normal is Minority (e.g. 40%), w_norm (1.25) > w_abnorm (0.83).
    # This maximizes Accuracy but hurts Abnormal Recall.
    # DECISION: We prioritize Recall. We will Flip the weights or Equalize them + Bias.
    # Let's set w_abnorm = w_norm * 1.5 (Aggressive Recall Bias)
    
    final_w_norm = 1.0
    final_w_abnorm = (num_normal / num_abnormal) * 1.5 # Bias towards Abnormal
    
    # Normalize
    scale = 1.0 / min(final_w_norm, final_w_abnorm)
    final_w_norm *= scale
    final_w_abnorm *= scale
    
    print(f"Final Weights for Training: Normal={final_w_norm:.2f}, Abnormal={final_w_abnorm:.2f}")
    
    return torch.tensor([final_w_norm, final_w_abnorm], dtype=torch.float32)

def train_gatekeeper():
    print("--- Starting Stage 1: The Gatekeeper (Normal vs Abnormal) ---")
    os.makedirs("models", exist_ok=True)
    
    # 1. Data
    print("Loading Dataset...")
    dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, detected_seq_len=5000)
    
    # Identify indices for Normal classes
    normal_indices = set()
    for name, idx in dataset.class_map.items():
        if name in ['NORM', 'Sinus_Rhythm']:
            normal_indices.add(idx)
            
    print(f"Normal Classes Indices: {normal_indices}")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model
    model = SEResNet34(num_classes=2, input_channels=NUM_LEADS)
    model = model.to(DEVICE)
    
    # 3. Weights calculation
    class_weights = compute_class_weights(dataset, normal_indices).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_recall = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Training
        pbar = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, original_labels in pbar:
            inputs = inputs.to(DEVICE)
            
            # Remap Labels (Vectorized)
            # Create a boolean mask: True if label is in normal_indices
            # Since normal_indices is a set of small ints, we can use a lookup tensor if needed, but python list comprehension is fast enough for batch 32
            # Optimized list comprehension
            binary_labels = torch.tensor(
                [0 if l.item() in normal_indices else 1 for l in original_labels], 
                device=DEVICE
            )
            
            optimizer.zero_grad()
            outputs = model(inputs)
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
                
                # Remap (Vectorized)
                binary_labels = torch.tensor(
                    [0 if l.item() in normal_indices else 1 for l in original_labels], 
                    device=DEVICE
                )
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(binary_labels.cpu().numpy())
        
        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0) # Specificity for Normal, Recall for Abnormal
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.2%} | Prec: {prec:.2%} | Recall (Sensitivity): {rec:.2%} | F1: {f1:.2%}")
        
        # Save Best by Recall (Primary Objective)
        if rec > best_recall:
            best_recall = rec
            torch.save(model.state_dict(), "models/stage1_gatekeeper.pth")
            print("  -> Saved Best Gatekeeper Model (Recall Optimized)")
        elif rec == best_recall and acc > 0: # Tie breaker
             pass # Logic can be added
             
    print(f"Stage 1 Training Complete. Best Recall: {best_recall:.2%}")

if __name__ == "__main__":
    train_gatekeeper()
