"""
Engine B — L3 Rhythm Specialist Training
=========================================
Multi-class SE-ResNet-34 classifier for rhythm disorders on PTB-XL at 500Hz.
Loss:       CrossEntropyLoss (stable for imbalanced multi-class)
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
from src.config import STAGE3_RHYTHM_DATA_PATH, STAGE3_RHYTHM_MODEL_PATH, SIGNAL_LENGTH
from src.engine_b_signal.train_signal_model import DATA_DIR, NUM_LEADS, group_diagnostic_classes
from src.engine_b_signal.models.resnet1d_se import SEResNet34
from src.prepare_stage2_data import CLASS_0_RHYTHM

# Config
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Rhythm Classes Mapping — deduplicate ('DIG'/'Dig' are the same)
RHYTHM_CLASSES = sorted(set(CLASS_0_RHYTHM))
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(RHYTHM_CLASSES)}




# =============================================================================
# Rhythm Dataset
# =============================================================================
class RhythmDataset(Dataset):
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
            print(f"  [RhythmDataset] Dropped {dropped} rows with missing signal files")

        self.data_frame = df.loc[valid_rows].reset_index(drop=True)

        # Pre-process labels
        self.labels = []
        for idx, row in self.data_frame.iterrows():
            label_str = group_diagnostic_classes(str(row['label']))
            if label_str not in CLASS_TO_IDX:
                label_str = group_diagnostic_classes(str(row.get('diagnostic_superclass', '')))
            if label_str in CLASS_TO_IDX:
                self.labels.append(CLASS_TO_IDX[label_str])
            else:
                self.labels.append(0)
        self.labels = np.array(self.labels)

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
            label_str = group_diagnostic_classes(str(row['label']))
            label = CLASS_TO_IDX.get(label_str, 0)
            return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        signal = np.load(file_path)

        if signal.shape[0] != 12 and signal.shape[1] == 12:
            signal = signal.T

        current_len = signal.shape[1]
        if current_len != self.seq_len:
            sig_torch = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
            sig_torch = torch.nn.functional.interpolate(sig_torch, size=self.seq_len, mode='linear')
            signal = sig_torch.squeeze(0).numpy()

        # Augmentation: Noise Injection + Time Shift (Mixup-style)
        if self.augment:
            noise = np.random.normal(0, 0.01, signal.shape)
            signal = signal + noise

        # Get Label
        label_str = group_diagnostic_classes(str(row['label']))
        if label_str not in CLASS_TO_IDX:
            label_str = group_diagnostic_classes(str(row.get('diagnostic_superclass', '')))
        label = CLASS_TO_IDX.get(label_str, 0)

        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# =============================================================================
# Training
# =============================================================================
def train_rhythm_specialist():
    print("=" * 60)
    print("Engine B — L3 Rhythm Specialist Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Classes ({len(RHYTHM_CLASSES)}): {RHYTHM_CLASSES}")

    # 1. Load Data
    full_df = pd.read_csv(STAGE3_RHYTHM_DATA_PATH)
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, random_state=42,
        stratify=full_df['router_label'] if 'router_label' in full_df else None
    )

    print(f"Train Size: {len(train_df)} | Val Size: {len(val_df)}")

    train_dataset = RhythmDataset(train_df, root_dir=DATA_DIR, augment=True)
    val_dataset = RhythmDataset(val_df, root_dir=DATA_DIR, augment=False)

    # Weighted Sampler
    train_labels = train_dataset.labels
    class_counts = np.bincount(train_labels, minlength=len(RHYTHM_CLASSES)).astype(float)
    print(f"Training Class Counts: {class_counts}")

    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Model
    model = SEResNet34(num_classes=len(RHYTHM_CLASSES), input_channels=NUM_LEADS)
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")

    # 3. Loss — CrossEntropyLoss (stable for imbalanced multi-class)
    criterion = nn.CrossEntropyLoss()

    # 4. Optimizer + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

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
            torch.save(model.state_dict(), STAGE3_RHYTHM_MODEL_PATH)
            print(f"  -> Saved Best Rhythm Specialist (Val Acc: {val_acc:.2%})")

    print(f"\nTraining Complete. Best Val Acc: {best_acc:.2%}")
    all_class_ids = list(range(len(RHYTHM_CLASSES)))
    print("\nFinal Classification Report:")
    print(classification_report(
        val_labels_list, val_preds,
        labels=all_class_ids,
        target_names=RHYTHM_CLASSES,
        zero_division=0
    ))


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_rhythm_specialist()
