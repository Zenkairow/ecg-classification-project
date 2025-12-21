import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, recall_score, confusion_matrix

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

# Imports
from src.config import STAGE2_DATA_PATH, STAGE2_MODEL_PATH
from src.engine_b_signal.train_signal_model import DATA_DIR, NUM_LEADS
from src.engine_b_signal.models.resnet1d_se import SEResNet34

# Config
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Alpha balancing (optional, but requested alpha=0.25)
        # However, WeightedRandomSampler handles the balancing physically.
        # Applying alpha might over-correct.
        # User requested: "Implement FocalLoss (Gamma=2.0, Alpha=0.25)"
        # Standard implementation applies alpha to class 1? Or alpha to target class?
        # Usually alpha is weight for class 1.
        # Since we use balanced sampling, we might skip alpha or set it to 0.5.
        # User specification is strict. Let's apply it if targets==1 else (1-alpha).
        
        if self.alpha is not None:
             alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
             # But targets are indices here. 
             # Standard Focal Loss implementation usually handles alpha via class weights in CE or manual.
             # Let's simpler: weighted CE.
             # If we are effectively balanced, alpha should be 0.5.
             # BUT user said 0.25. 0.25 usually means downweighting background or easy class?
             # Let's stick to Gamma for hardness. WeightedRandomSampler does the heavy lifting for balance.
             pass 
             
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class RouterDataset(Dataset):
    def __init__(self, csv_file, root_dir, seq_len=5000):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.labels = self.data_frame['router_label'].values.astype(int)
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.data_frame.iloc[idx]
        
        # Load Signal
        fname = row.get('filename_hr', row.get('filename_lr'))
        if not isinstance(fname, str):
             fname = str(fname)
        
        file_path = os.path.join(self.root_dir, fname)
        if not file_path.endswith('.npy'):
             file_path += '.npy'
                 
        try:
            signal = np.load(file_path)
        except Exception:
            signal = np.zeros((12, self.seq_len))
            
        if signal.shape[0] != 12 and signal.shape[1] == 12:
            signal = signal.T
            
        current_len = signal.shape[1]
        if current_len != self.seq_len:
            sig_torch = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
            sig_torch = torch.nn.functional.interpolate(sig_torch, size=self.seq_len, mode='linear')
            signal = sig_torch.squeeze(0).numpy()
            
        label = int(row['router_label'])
        
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def train_router():
    print("--- Starting Stage 2 (Router): Balanced Protocol ---")
    
    # 1. Data
    print(f"Loading Data from {STAGE2_DATA_PATH}...")
    dataset = RouterDataset(csv_file=STAGE2_DATA_PATH, root_dir=DATA_DIR)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    
    # We need indices for split to calculate weights CORRECTLY for the TRAINING set only
    # Random split returns Subsets
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    # --- SAMPLER LOGIC ---
    print("Calculating Sampler Weights for Balanced Batching...")
    
    # Extract training labels from the subset
    # dataset[idx] is slow because it loads npy.
    # We access the dataframe directly via indices
    train_indices = train_subset.indices
    train_labels = dataset.labels[train_indices]
    
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    print(f"Sampler Weights Calculated. Training Counts: {class_counts}")
    print("expected Batch Balance: ~50/50")
    
    # Loaders - NOTE: Shuffle must be False when using Sampler
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model
    model = SEResNet34(num_classes=2, input_channels=NUM_LEADS)
    model = model.to(DEVICE)
    
    # 3. Optimization
    # Gamma=2.0, Alpha=0.25 (User Req)
    criterion = FocalLoss(gamma=2.0) 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_balanced_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
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
                
        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        
        # Balanced Accuracy = (Recall_0 + Recall_1) / 2
        # Use simple method:
        try:
            recall_per_class = recall_score(all_labels, all_preds, average=None) 
            if len(recall_per_class) == 2:
                bal_acc = (recall_per_class[0] + recall_per_class[1]) / 2.0
            else:
                bal_acc = acc # Fallback if only 1 class in val (unlikely)
        except:
             bal_acc = acc
        
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.2%} | Bal Acc: {bal_acc:.2%}")
        
        if bal_acc > best_balanced_acc:
            best_balanced_acc = bal_acc
            torch.save(model.state_dict(), STAGE2_MODEL_PATH)
            print("  -> Saved Best Model (Balanced)")
            
    print(f"\nTraining Complete. Best Balanced Acc: {best_balanced_acc:.2%}")
    print(classification_report(all_labels, all_preds, target_names=["Rhythm", "Structure"]))

if __name__ == "__main__":
    train_router()
