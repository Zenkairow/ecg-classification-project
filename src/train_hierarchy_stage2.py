import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

# Imports
from src.engine_b_signal.train_signal_model import ECGSignalDataset, CSV_PATH, DATA_DIR, NUM_LEADS
from src.engine_b_signal.models.resnet1d_se import SEResNet34

# Config
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Router Logic ---
RHYTHM_CLASSES = [
    'Atrial_Fibrillation', 'SVT', 'PVC', 'Paced', 'WPW', 
    'AV_Block_1st_Deg', 'AV_Block_2nd_Deg', 'AV_Block_3rd_Deg', 
    'RBBB', 'LBBB', 'Fascicular_Block', 'IVCD', 'LNGQT'
]

STRUCTURE_CLASSES = [
    'MI_Anterior', 'MI_Inferior', 'MI_Lateral', 
    'Ischemia', 'NonSpecific_ST', 
    'Left_Hypertrophy', 'Right_Hypertrophy', 'DIG'
    # 'EL', 'NDT' treated as ignore/noise or mapped to Structure? 
    # Let's map NDT/EL to Structure for now so they go to Morphology specialist (which sees shapes)
]

def get_router_label(label_name):
    """
    Returns:
        0 for Rhythm/Conduction
        1 for Structure/Perfusion
        None for Ignored (Normal)
    """
    if label_name in ['NORM', 'Sinus_Rhythm']:
        return None # Ignore Normal samples for this stage
    
    if label_name in RHYTHM_CLASSES:
        return 0
    
    # Default to Structure for everything else (MI, ST, etc)
    return 1 

def collate_fn_filter_normal(batch):
    """
    Custom collate to filter out Normal samples on the fly.
    """
    filtered_inputs = []
    filtered_labels = []
    
    for item in batch:
        # dataset[i] returns (signal, label_idx)
        # We need to map label_idx -> name -> router_label
        sig, label_idx = item
        # We need the class map. 
        # Hack: we will assume we can access it or pass it.
        # Actually, let's do filtering in the loop to avoid batch issues (variable size).
        # Better: Filter indices in Dataset? No, dataset assumes full CSV.
        # We will handle it in the Training Loop: if label is None, skip.
        # But for Batching, we need valid batches.
        pass
    
    # Standard collate
    return torch.utils.data.dataloader.default_collate(batch)

def train_router():
    print("--- Starting Stage 2: The Router (Rhythm vs Structure) ---")
    os.makedirs("models", exist_ok=True)
    
    # 1. Data
    print("Loading Dataset...")
    full_dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, detected_seq_len=5000)
    
    # Create a Filtered Subset (Only Abnormals)
    # This ensures epochs are clean and batches are full.
    print("Filtering for Abnormal Samples...")
    
    abnormal_indices = []
    
    # Invert map
    idx_to_class = {v: k for k, v in full_dataset.class_map.items()}
    
    # Iterate all annotations to find indices
    loss_col = 'label' if 'label' in full_dataset.annotations.columns else 'diagnostic_superclass'
    # Access internal dataframe to speed up
    labels = full_dataset.annotations[loss_col].values
    
    # Need access to group logic or just raw string matching
    # Let's use the helper we used before
    from src.engine_b_signal.train_signal_model import group_diagnostic_classes
    
    processed_labels = [] # To store the binary 0/1 for the subset
    
    for i, raw_label in enumerate(labels):
        group = group_diagnostic_classes(raw_label)
        router_lbl = get_router_label(group)
        
        if router_lbl is not None:
            abnormal_indices.append(i)
            processed_labels.append(router_lbl)
            
    print(f"Total Samples: {len(full_dataset)}")
    print(f"Abnormal Samples (Stage 2 Data): {len(abnormal_indices)}")
    
    # Create Subset
    subset = torch.utils.data.Subset(full_dataset, abnormal_indices)
    
    # We need to feed the NEW binary labels (processed_labels) not the old multi-class ones.
    # We can create a custom wrapper dataset.
    
    class RouterDataset(torch.utils.data.Dataset):
        def __init__(self, original_subset, new_labels):
            self.subset = original_subset
            self.new_labels = new_labels
            
        def __len__(self):
            return len(self.subset)
            
        def __getitem__(self, idx):
            # Subset[idx] returns (signal, original_label)
            signal, _ = self.subset[idx]
            target = self.new_labels[idx]
            return signal, target
            
    router_dataset = RouterDataset(subset, processed_labels)
    
    # Split
    train_size = int(0.8 * len(router_dataset))
    val_size = len(router_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(router_dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model
    model = SEResNet34(num_classes=2, input_channels=NUM_LEADS)
    model = model.to(DEVICE)
    
    # 3. Optimization
    # Check balance
    rhythm_count = sum([1 for x in processed_labels if x == 0])
    struct_count = sum([1 for x in processed_labels if x == 1])
    print(f"Balance: Rhythm={rhythm_count}, Structure={struct_count}")
    
    # Standard weighted loss if needed
    weights = torch.tensor([1.0, rhythm_count / struct_count]).to(DEVICE) # Basic balancing
    print(f"Using Weights: {weights}")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Stage 2 Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Validation Acc: {acc:.2%}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/stage2_router.pth")
            print("  -> Saved Best Router Model")
            
    print(f"Stage 2 Training Complete. Best Acc: {best_acc:.2%}")
    # Final Report
    print("\nClassification Report (Val Set):")
    print(classification_report(all_labels, all_preds, target_names=["Rhythm", "Structure"]))

if __name__ == "__main__":
    train_router()
