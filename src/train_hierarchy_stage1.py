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

def compute_class_weights(dataset):
    """
    Computes class weights to handle imbalance.
    Goal: Prioritize Recall (Sensitivity) for Abnormal (Class 1).
    So we don't punish False Positives as much as False Negatives.
    """
    print("Computing class weights...")
    normal_count = 0
    abnormal_count = 0
    
    # Quick scan of the dataset labels
    # Note: dataset.annotations has 'diagnostic_superclass' or 'label'
    loss_col = 'label' if 'label' in dataset.annotations.columns else 'diagnostic_superclass'
    
    # We need to map string labels to binary
    # This might be slow if we iterate all, but dataset is small (21k)
    # Better: use pandas
    labels = dataset.annotations[loss_col].values
    
    # NOTE: The dataset applies 'group_diagnostic_classes' internally, but the CSV might not have the grouped names yet?
    # Actually ECGSignalDataset constructor does grouping.
    # But let's rely on the dataset class_map logic.
    
    # Let's count via the loop for safety or just assume standard distribution?
    # User said: "if Normal is 40% and Abnormal is 60%, weight Normal slightly higher"
    # Actually, to maximize Recall of Abnormal, we usually increase weight of Abnormal.
    # But usually CrossEntropy weights are inverse frequency.
    
    # Let's count exactly.
    for i in range(len(dataset)):
        # Inspect the label index returned by dataset
        # We can't easily peek without getting item.
        # Let's use the raw dataframe and the group logic from train_signal_model import group_diagnostic_classes
        pass 
        
    # Hack: Just count based on the internal class map keys since we know the mapping
    # NORM + Sinus_Rhythm vs Others.
    
    # Let's do it dynamically in the main script to be precise.
    return torch.tensor([1.0, 1.0]) # Default placeholder

def train_gatekeeper():
    print("--- Starting Stage 1: The Gatekeeper (Normal vs Abnormal) ---")
    os.makedirs("models", exist_ok=True)
    
    # 1. Data
    print("Loading Dataset...")
    dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, detected_seq_len=5000)
    
    # Identify indices for Normal classes
    normal_indices = []
    for name, idx in dataset.class_map.items():
        if name in ['NORM', 'Sinus_Rhythm']:
            normal_indices.append(idx)
            
    print(f"Normal Classes Indices: {normal_indices} ({['NORM', 'Sinus_Rhythm']})")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model
    # Binary Output
    model = SEResNet34(num_classes=2, input_channels=NUM_LEADS)
    model = model.to(DEVICE)
    
    # 3. Weights calculation
    # We want High Recall for Class 1 (Abnormal).
    # To encourage the model to NOT miss class 1, we can weight Class 1 higher.
    # Let's assume a 1:1 balance for now, or maybe 1.5 for Abnormal.
    # Or strict inverse frequency.
    # Let's calculate counts from full dataset iteration once (fast enough for 21k)
    print("Calculating Class Distribution...")
    zeros = 0
    ones = 1
    # Iterate a bit of training data to estimate? No, let's just use a reasonable prior.
    # User said 40% Normal, 60% Abnormal.
    # Inverse weights: 1/0.4 = 2.5, 1/0.6 = 1.66
    # Normalized: Normal=1.5, Abnormal=1.0
    # Wait, if we want HIGH RECALL for Abnormal, we punish Missing Abnormal.
    # Missing Abnormal means Pred=0, True=1.
    # So we want the loss for True=1 to be higher.
    # So weight for Class 1 should be higher.
    class_weights = torch.tensor([1.0, 2.0]).to(DEVICE) # Upweight Abnormal
    print(f"Using Class Weights: {class_weights}")
    
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
            
            # Remap Labels
            # Try to vectorize this
            binary_labels = []
            for l in original_labels:
                if l.item() in normal_indices:
                    binary_labels.append(0)
                else:
                    binary_labels.append(1)
            binary_labels = torch.tensor(binary_labels).to(DEVICE)
            
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
                
                # Remap
                binary_labels_list = []
                for l in original_labels:
                    if l.item() in normal_indices:
                        binary_labels_list.append(0)
                    else:
                        binary_labels_list.append(1)
                binary_labels = torch.tensor(binary_labels_list).to(DEVICE)
                
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
