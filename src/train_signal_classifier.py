import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from model_v6 import ECGTransformer

# --- Configuration ---
BATCH_SIZE = 32  # Signals are lighter than images, can increase BS
SEQ_LEN = 1000   # 10 seconds @ 100Hz
NUM_LEADS = 12
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50  # Transformers need more epochs
DATA_DIR = 'data_synthesis/output/output/signals/' # Assumed path for .npy files
CSV_PATH = 'data/train_labels.csv'
MODEL_SAVE_PATH = 'models/signal_transformer_v6.pth'

# --- Dataset Class ---
class ECGSignalDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.root_dir = root_dir
        
        # Load CSV
        try:
            self.annotations = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            sys.exit(1)
            
        # Class Mapping
        loss_col = 'label' if 'label' in self.annotations.columns else 'diagnostic_superclass'
        unique_labels = sorted(self.annotations[loss_col].unique().tolist())
        self.class_map = {label: i for i, label in enumerate(unique_labels)}
        
        print(f"Signal Dataset Loaded. Total samples: {len(self.annotations)}")
        print(f"Detected {len(unique_labels)} Classes found.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 1. Get Filename (Convert .png to .npy if needed)
        # CSV has 'sample_123.png'. We need 'sample_123.npy'
        base_name = str(self.annotations.iloc[index]['filename'])
        if base_name.endswith('.png'):
            base_name = base_name.replace('.png', '.npy')
        elif not base_name.endswith('.npy'):
            base_name = base_name + '.npy'
            
        signal_path = os.path.join(self.root_dir, base_name)
        
        # 2. Load Signal
        try:
            # Expecting shape [12, 1000] or [1000, 12]
            signal = np.load(signal_path)
            
            # Ensure shape is [12, 1000] for our model
            if signal.shape[0] != 12 and signal.shape[1] == 12:
                signal = signal.T 
                
            # Basic Normalization (Z-score)
            # Avoid division by zero
            mean = np.mean(signal, axis=1, keepdims=True)
            std = np.std(signal, axis=1, keepdims=True)
            std[std == 0] = 1.0
            signal = (signal - mean) / std
            
            # Tensor conversion
            signal_tensor = torch.from_numpy(signal).float()
            
        except FileNotFoundError:
            # print(f"Warning: Missing signal file {signal_path}")
            # Return dummy zero signal
            signal_tensor = torch.zeros((NUM_LEADS, SEQ_LEN)).float()
        except Exception as e:
            print(f"Error loading {signal_path}: {e}")
            signal_tensor = torch.zeros((NUM_LEADS, SEQ_LEN)).float()
            
        # 3. Get Label
        loss_col = 'label' if 'label' in self.annotations.columns else 'diagnostic_superclass'
        label_str = self.annotations.iloc[index][loss_col]
        label = self.class_map[label_str]

        return signal_tensor, label

# --- Training Function ---
def train_model():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR)
    
    if len(dataset) == 0:
        print("CRITICAL: Dataset is empty!")
        return

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training on {train_size} signals, Validating on {val_size}")

    # Model (v6 Transformer)
    print("Initializing ECGTransformer (v6)...")
    num_classes = len(dataset.class_map)
    model = ECGTransformer(num_classes=num_classes, input_channels=NUM_LEADS)
    model = model.to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Loop
    best_val_acc = 0.0
    print(f"Starting Signal training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc="Training")
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)
            
            # Forward
            scores = model(data)
            loss = criterion(scores, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predictions = scores.max(1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
            loop.set_postfix(loss=loss.item())
            
        epoch_acc = correct / total if total > 0 else 0
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Results -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                scores = model(data)
                loss = criterion(scores, targets)
                val_loss += loss.item()
                
                _, predictions = scores.max(1)
                val_correct += (predictions == targets).sum().item()
                val_total += targets.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Acc: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New Best Signal Model! Saving to {MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_model()
