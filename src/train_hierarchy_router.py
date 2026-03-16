"""
Engine B — L2 Router Training (Rhythm vs Structure)
====================================================
Binary SE-ResNet-34 classifier on PTB-XL at 500Hz.
Loss:       Focal Loss (gamma=2.0)
Optimizer:  AdamW (weight_decay=1e-2)
Scheduler:  ReduceLROnPlateau
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

# Imports
from src.config import STAGE2_DATA_PATH, STAGE2_MODEL_PATH, SIGNAL_LENGTH
from src.engine_b_signal.train_signal_model import DATA_DIR, NUM_LEADS
from src.engine_b_signal.models.resnet1d_se import SEResNet34

# Config
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # Binary: 0=Rhythm, 1=Structure


# =============================================================================
# Focal Loss
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# =============================================================================
# Router Dataset
# =============================================================================
class RouterDataset(Dataset):
    def __init__(self, df, root_dir, seq_len=5000, augment=False):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.augment = augment

        # Pre-filter: remove rows whose signal files don't exist
        valid_rows = []
        for idx, row in df.iterrows():
            fname = self._resolve_filename(row)
            fpath = os.path.join(root_dir, fname)
            if not fpath.endswith('.npy'):
                fpath += '.npy'
            if os.path.exists(fpath):
                valid_rows.append(idx)

        dropped = len(df) - len(valid_rows)
        if dropped > 0:
            print(f"  [RouterDataset] Dropped {dropped} rows with missing signal files")

        self.data_frame = df.loc[valid_rows].reset_index(drop=True)
        self.labels = self.data_frame['router_label'].values.astype(int)

    def _resolve_filename(self, row):
        fname = str(row.get('filename', ''))
        if pd.isna(fname) or fname == 'nan' or fname == '':
            ecg_id = row.get('ecg_id', '')
            fname = f"sample_{ecg_id}.npy"
        return fname

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        fname = self._resolve_filename(row)
        file_path = os.path.join(self.root_dir, fname)

        if not os.path.exists(file_path):
            if not file_path.endswith('.npy'):
                file_path += '.npy'

        if not os.path.exists(file_path):
            signal = np.zeros((12, self.seq_len), dtype=np.float32)
            label = int(row['router_label'])
            return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        signal = np.load(file_path)

        if signal.shape[0] != 12 and signal.shape[1] == 12:
            signal = signal.T

        current_len = signal.shape[1]
        if current_len != self.seq_len:
            sig_torch = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
            sig_torch = torch.nn.functional.interpolate(sig_torch, size=self.seq_len, mode='linear')
            signal = sig_torch.squeeze(0).numpy()

        if self.augment:
            noise = np.random.normal(0, 0.01, signal.shape)
            signal = signal + noise

        label = int(row['router_label'])
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# =============================================================================
# Training
# =============================================================================
def train_router():
    print("=" * 60)
    print("Engine B — L2 Router Training (Rhythm vs Structure)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Classes: ['Rhythm (0)', 'Structure (1)']")

    # 1. Load Data
    full_df = pd.read_csv(STAGE2_DATA_PATH)
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, random_state=42,
        stratify=full_df['router_label']
    )

    print(f"Train Size: {len(train_df)} | Val Size: {len(val_df)}")

    train_dataset = RouterDataset(train_df, root_dir=DATA_DIR, augment=True)
    val_dataset = RouterDataset(val_df, root_dir=DATA_DIR, augment=False)

    # Weighted Sampler
    train_labels = train_dataset.labels
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
    print(f"Training Class Counts: {class_counts}")

    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Model
    model = SEResNet34(num_classes=NUM_CLASSES, input_channels=NUM_LEADS)
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")

    # 3. Loss — Focal Loss
    loss_weight = torch.tensor(class_weights / class_weights.sum() * NUM_CLASSES, dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(gamma=2.0, weight=loss_weight)

    # 4. Optimizer + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []

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

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels_list, val_preds)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:02d} | Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), STAGE2_MODEL_PATH)
            print(f"  -> Saved Best Router (Val Acc: {val_acc:.2%})")

    print(f"\nTraining Complete. Best Val Acc: {best_acc:.2%}")
    print("\nFinal Classification Report:")
    print(classification_report(
        val_labels_list, val_preds,
        target_names=['Rhythm', 'Structure'],
        zero_division=0
    ))


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_router()
