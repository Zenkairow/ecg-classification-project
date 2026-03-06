"""
V3.0 Structure Specialist Training — AdvancedCardiacNet Backbone
================================================================
Loss:       Class-Balanced Focal Loss (gamma=2.0, per-class inverse-freq alpha)
Optimizer:  AdamW + Lookahead (k=5, alpha=0.5)
Scheduler:  CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import sys
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

# Imports
from src.config import STAGE3_STRUCTURE_DATA_PATH, V3_STRUCTURE_MODEL_PATH, SIGNAL_LENGTH
from src.engine_b_signal.train_signal_model import DATA_DIR, NUM_LEADS, group_diagnostic_classes
from src.models import AdvancedCardiacNet
from src.prepare_stage2_data import CLASS_1_STRUCTURE

# Reuse StructureDataset from existing specialist
from src.train_specialist_structure import StructureDataset

# Config
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STRUCTURE_CLASSES = sorted(CLASS_1_STRUCTURE)
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(STRUCTURE_CLASSES)}


# =============================================================================
# Class-Balanced Focal Loss
# =============================================================================

class ClassBalancedFocalLoss(nn.Module):
    """
    Focal Loss with per-class inverse-frequency alpha weights.
    Forces convergence on rare structural classes (e.g., Right_Hypertrophy).
    
    Args:
        gamma (float): Focusing parameter. Higher = more focus on hard examples.
        class_counts (ndarray): Per-class sample counts from training set.
        num_classes (int): Total number of classes.
    """
    def __init__(self, gamma=2.0, class_counts=None, num_classes=11):
        super().__init__()
        self.gamma = gamma
        
        if class_counts is not None:
            # Inverse frequency weighting, normalized to sum to num_classes
            weights = 1.0 / (class_counts + 1e-6)
            weights = weights / weights.sum() * num_classes
            self.register_buffer('alpha', torch.tensor(weights, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# =============================================================================
# Lookahead Optimizer Wrapper
# =============================================================================

class Lookahead(optim.Optimizer):
    """
    Lookahead optimizer wrapper (Zhang et al., 2019).
    Maintains slow weights updated every k steps.
    
    Args:
        base_optimizer: Inner optimizer (e.g., AdamW).
        k (int): Slow weight update frequency.
        alpha (float): Interpolation factor for slow weights.
    """
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        
        # Cache slow weights
        self.slow_weights = []
        for group in self.base_optimizer.param_groups:
            slow_group = []
            for p in group['params']:
                if p.requires_grad:
                    slow_group.append(p.data.clone())
            self.slow_weights.append(slow_group)
        
        # Satisfy Optimizer base class
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.defaults = self.base_optimizer.defaults

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self._step_count += 1
        
        if self._step_count % self.k == 0:
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                param_idx = 0
                for p in group['params']:
                    if p.requires_grad:
                        slow = self.slow_weights[group_idx][param_idx]
                        slow.add_(self.alpha * (p.data - slow))
                        p.data.copy_(slow)
                        param_idx += 1
        
        return loss

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @property
    def param_groups_prop(self):
        return self.base_optimizer.param_groups


# =============================================================================
# Training Function
# =============================================================================

def train_v3_structure():
    print("=" * 60)
    print("V3.0 Structure Specialist — AdvancedCardiacNet")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Classes ({len(STRUCTURE_CLASSES)}): {STRUCTURE_CLASSES}")
    
    # 1. Data Loading
    full_df = pd.read_csv(STAGE3_STRUCTURE_DATA_PATH)
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, random_state=42, 
        stratify=full_df['router_label'] if 'router_label' in full_df else None
    )
    
    print(f"Train Size: {len(train_df)} | Val Size: {len(val_df)}")
    
    train_dataset = StructureDataset(train_df, root_dir=DATA_DIR, augment=True)
    val_dataset = StructureDataset(val_df, root_dir=DATA_DIR, augment=False)
    
    # Sampler: Weighted for class balance
    train_labels = train_dataset.labels
    class_counts = np.bincount(train_labels, minlength=len(STRUCTURE_CLASSES)).astype(float)
    print(f"Training Class Counts: {class_counts}")
    
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model — AdvancedCardiacNet
    model = AdvancedCardiacNet(num_classes=len(STRUCTURE_CLASSES), input_channels=NUM_LEADS)
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} total | {trainable_params:,} trainable")
    
    # 3. Loss — Class-Balanced Focal Loss
    criterion = ClassBalancedFocalLoss(
        gamma=2.0, 
        class_counts=class_counts, 
        num_classes=len(STRUCTURE_CLASSES)
    ).to(DEVICE)
    
    # 4. Optimizer — AdamW + Lookahead
    base_optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    
    # 5. Scheduler — Cosine Annealing Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(base_optimizer, T_0=10, T_mult=2)
    
    best_acc = 0.0
    
    print(f"\nStarting Training for {EPOCHS} epochs...")
    print("-" * 60)
    
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
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        current_lr = base_optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} | Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | LR: {current_lr:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), V3_STRUCTURE_MODEL_PATH)
            print(f"  -> Saved Best Model (Val Acc: {val_acc:.2%})")
    
    print("\n" + "=" * 60)
    print(f"Training Complete. Best Val Acc: {best_acc:.2%}")
    print("=" * 60)
    
    # Final Classification Report
    all_class_ids = list(range(len(STRUCTURE_CLASSES)))
    print("\nFinal Validation Classification Report:")
    print(classification_report(
        val_labels, val_preds, 
        labels=all_class_ids, 
        target_names=STRUCTURE_CLASSES, 
        zero_division=0
    ))


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_v3_structure()
