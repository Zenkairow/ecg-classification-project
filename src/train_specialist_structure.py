import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

# Imports
from src.config import STAGE3_STRUCTURE_DATA_PATH, STAGE3_STRUCTURE_MODEL_PATH, SIGNAL_LENGTH
from src.engine_b_signal.train_signal_model import DATA_DIR, NUM_LEADS, group_diagnostic_classes
from src.engine_b_signal.models.resnet1d_se import SEResNet34
from src.prepare_stage2_data import CLASS_1_STRUCTURE

# Config
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Structure Classes Mapping (Local to this Specialist)
STRUCTURE_CLASSES = sorted(CLASS_1_STRUCTURE)
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(STRUCTURE_CLASSES)}

class StructureDataset(Dataset):
    def __init__(self, csv_file, root_dir, seq_len=5000):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.labels = []
        
        # Pre-process labels for sampler
        for idx, row in self.data_frame.iterrows():
             label_str = group_diagnostic_classes(str(row['label']))
             if label_str not in CLASS_TO_IDX:
                 label_str = group_diagnostic_classes(str(row['diagnostic_superclass']))
                 
             if label_str in CLASS_TO_IDX:
                 self.labels.append(CLASS_TO_IDX[label_str])
             else:
                 # Default logic (should be rare if data prep worked)
                 self.labels.append(0) 
                 
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.data_frame.iloc[idx]
        
        # Load Signal
        fname = row.get('filename_hr')
        if pd.isna(fname) or fname == 'nan' or fname == '':
             fname = str(row.get('filename_hr', ''))
        
        file_path = os.path.join(self.root_dir, fname)
        
        if not os.path.exists(file_path):
             if not file_path.endswith('.npy'):
                  file_path += '.npy'
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Signal missing: {file_path}")
            
        signal = np.load(file_path)
        
        if signal.shape[0] != 12 and signal.shape[1] == 12:
            signal = signal.T
            
        current_len = signal.shape[1]
        if current_len != self.seq_len:
            sig_torch = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
            sig_torch = torch.nn.functional.interpolate(sig_torch, size=self.seq_len, mode='linear')
            signal = sig_torch.squeeze(0).numpy()
            
        # Get Label
        label_str = group_diagnostic_classes(str(row['label']))
        if label_str not in CLASS_TO_IDX:
             label_str = group_diagnostic_classes(str(row['diagnostic_superclass']))
             
        label = CLASS_TO_IDX.get(label_str, 0)
        
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def train_structure_specialist():
    print("--- Starting Stage 3: Structure Specialist ---")
    print(f"Classes ({len(STRUCTURE_CLASSES)}): {STRUCTURE_CLASSES}")
    
    # 1. Data
    dataset = StructureDataset(csv_file=STAGE3_STRUCTURE_DATA_PATH, root_dir=DATA_DIR)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    # Sampler
    print("Calculating Sampler Weights...")
    train_indices = train_subset.indices
    train_labels = dataset.labels[train_indices]
    
    class_counts = np.bincount(train_labels, minlength=len(STRUCTURE_CLASSES))
    print(f"Training Class Counts: {class_counts}")
    
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[train_labels]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model
    model = SEResNet34(num_classes=len(STRUCTURE_CLASSES), input_channels=NUM_LEADS)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        scheduler.step()
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), STAGE3_STRUCTURE_MODEL_PATH)
            print("  -> Saved Best Model")
            
    print(f"\nTraining Complete. Best Val Acc: {best_acc:.2%}")
    all_class_ids = list(range(len(STRUCTURE_CLASSES)))
    print(classification_report(val_labels, val_preds, labels=all_class_ids, target_names=STRUCTURE_CLASSES, zero_division=0))

if __name__ == "__main__":
    train_structure_specialist()
