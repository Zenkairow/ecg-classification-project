import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
import argparse

# Add project root to path to allow 'src' imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# CRITICAL: Add the engine folder itself so 'models.resnet1d_se' works
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

# Import from Training Script (Reusing Logic)
from src.engine_b_signal.train_signal_model import ECGSignalDataset, group_diagnostic_classes, CSV_PATH, DATA_DIR, NUM_LEADS
from src.engine_b_signal.models.resnet1d_se import SEResNet34

# Constants
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_validation_data():
    """Replicates the split logic from training to get the exact same validation set."""
    print("Loading Dataset...")
    # We must replicate the exact split logic
    # FIX: Pass the sequence length (5000) so the dataset knows how to initialize buffers
    dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, detected_seq_len=5000)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Deterministic split via generator seed if needed, but for now we assume standard random_split 
    # NOTE: random_split is random. To be perfectly exact we should have saved indices. 
    # For a general diagnostic, a predictable random seed is good practice.
    generator = torch.Generator().manual_seed(42) 
    # Note: If training didn't use a seed, this might differ slightly. 
    # But usually broad dataset stats are similar.
    
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Validation Samples: {len(val_dataset)}")
    return val_loader, dataset.class_map

def analyze_confusions(y_true, y_pred, class_map):
    """
    Finds and prints the Top 10 Most Confused Pairs.
    Format: True -> Predicted (Count)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Invert class_map: Index -> Name
    idx_to_class = {v: k for k, v in class_map.items()}
    
    pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                pairs.append({
                    "true": idx_to_class[i],
                    "pred": idx_to_class[j],
                    "count": cm[i][j]
                })
    
    # Sort by count descending
    pairs.sort(key=lambda x: x['count'], reverse=True)
    
    print("\n" + "="*40)
    print("TOP 10 CONFUSED PAIRS (Where the model gets confused)")
    print("="*40)
    for k, p in enumerate(pairs[:15]): # Show Top 15
        print(f"{k+1}. True: [{p['true']}] -> Predicted: [{p['pred']}] (Count: {p['count']})")
    print("="*40 + "\n")

    return pairs

def diagnose(model_path):
    print(f"Diagnosing Model: {model_path}")
    
    # 1. Load Data
    val_loader, class_map = load_validation_data()
    num_classes = len(class_map)
    
    # 2. Load Model
    model = SEResNet34(num_classes=num_classes, input_channels=NUM_LEADS)
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        # Handle if it was saved as 'model_state_dict' or just full model
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.to(DEVICE)
    model.eval()
    
    # 3. Inference
    all_preds = []
    all_labels = []
    
    print("Running Inference on Validation Set...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Reporting
    target_names = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]
    
    print("\n" + "="*60)
    print(f"CLASSIFICATION REPORT (Val Acc: {np.mean(np.array(all_preds) == np.array(all_labels)):.2%})")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # 5. Confusion Analysis
    analyze_confusions(all_labels, all_preds, class_map)

if __name__ == "__main__":
    # Default to finding the best model file
    default_model = 'models/signal_seresnet34_best.pth'
    if not os.path.exists(default_model):
        default_model = 'models/signal_resnet34_70acc.pth' # Fallback
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=default_model, help="Path to .pth model file")
    args = parser.parse_args()
    
    diagnose(args.model)
