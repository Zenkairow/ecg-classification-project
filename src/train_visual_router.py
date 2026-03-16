"""
V3.0 Visual Router & Specialist Training
=========================================
Modes:
  --mode router     : Train ResNet-50 binary classifier (Rhythm=0 vs Structure=1)
  --mode rhythm     : Fine-tune rhythm specialist on rhythm subset
  --mode structure  : Fine-tune structure specialist on structure subset

Uses Smart Preprocessing Pipeline (Cropping + CLAHE) from ECGScanner.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.config import (
    VISUAL_ROUTER_DATA_PATH, VISUAL_RHYTHM_DATA_PATH, VISUAL_STRUCTURE_DATA_PATH,
    VISUAL_ROUTER_MODEL_PATH, VISUAL_RHYTHM_MODEL_PATH, VISUAL_STRUCTURE_MODEL_PATH,
)
from src.engine_a_visual.preprocess import ECGScanner
from src.engine_a_visual.train_visual_model import DATA_DIR as VISUAL_DATA_DIR
from src.prepare_visual_hierarchy import group_visual_label, CLASS_0_RHYTHM, CLASS_1_STRUCTURE

# Config
BATCH_SIZE = 4       # Halved to fit 1024x1024 in VRAM
IMAGE_SIZE = 1024    # Doubled for maximum ECG grid detail
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# =============================================================================
# Visual ECG Dataset with Smart Preprocessing
# =============================================================================

class VisualHierarchyDataset(Dataset):
    """
    ECG Image dataset with Smart Preprocessing (Crop + CLAHE).
    Supports router (binary), rhythm specialist, and structure specialist modes.
    """
    def __init__(self, csv_path, image_dir, mode='router', transform=None):
        self.image_dir = image_dir
        self.mode = mode
        self.transform = transform
        self.scanner = ECGScanner(target_size=(IMAGE_SIZE, IMAGE_SIZE))
        
        df = pd.read_csv(csv_path)
        
        # Build label mapping based on mode
        if mode == 'router':
            # Binary: 0=Rhythm, 1=Structure
            self.labels = df['router_label'].values.astype(int)
            self.class_names = ['Rhythm', 'Structure']
        elif mode == 'rhythm':
            # Multi-class rhythm specialist
            grouped = df['grouped_label'].apply(str)
            unique = sorted(grouped.unique().tolist())
            self.class_map = {name: idx for idx, name in enumerate(unique)}
            self.labels = grouped.map(self.class_map).values.astype(int)
            self.class_names = unique
        elif mode == 'structure':
            # Multi-class structure specialist
            grouped = df['grouped_label'].apply(str)
            unique = sorted(grouped.unique().tolist())
            self.class_map = {name: idx for idx, name in enumerate(unique)}
            self.labels = grouped.map(self.class_map).values.astype(int)
            self.class_names = unique
        
        self.filenames = df['filename'].values
        self.num_classes = len(set(self.labels))
        
        print(f"[{mode.upper()}] Loaded {len(self.filenames)} images, {self.num_classes} classes")
        print(f"  Classes: {self.class_names if hasattr(self, 'class_names') else 'N/A'}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = str(self.filenames[idx])
        img_path = os.path.join(self.image_dir, fname)
        
        try:
            # Smart Preprocessing: Crop + CLAHE + Resize
            processed_pil = self.scanner.preprocess(img_path)
        except Exception:
            # Fallback: load with PIL directly
            try:
                processed_pil = Image.open(img_path).convert("RGB")
                processed_pil = processed_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
            except Exception:
                processed_pil = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        
        if self.transform:
            processed_pil = self.transform(processed_pil)
        
        label = int(self.labels[idx])
        return processed_pil, torch.tensor(label, dtype=torch.long)


# =============================================================================
# Training Function
# =============================================================================

def train_visual(mode='router', epochs=20):
    print("=" * 60)
    print(f"V3.0 Visual {'Router' if mode == 'router' else mode.capitalize() + ' Specialist'} Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Select paths by mode
    if mode == 'router':
        csv_path = VISUAL_ROUTER_DATA_PATH
        model_save_path = VISUAL_ROUTER_MODEL_PATH
    elif mode == 'rhythm':
        csv_path = VISUAL_RHYTHM_DATA_PATH
        model_save_path = VISUAL_RHYTHM_MODEL_PATH
    elif mode == 'structure':
        csv_path = VISUAL_STRUCTURE_DATA_PATH
        model_save_path = VISUAL_STRUCTURE_MODEL_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        print("Run src/prepare_visual_hierarchy.py first.")
        return
    
    # Transforms (Medical-safe augmentations)
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load full dataset (no transform yet — will apply via SubsetWrapper)
    full_dataset = VisualHierarchyDataset(csv_path, VISUAL_DATA_DIR, mode=mode, transform=None)
    num_classes = full_dataset.num_classes
    
    if len(full_dataset) == 0:
        print("CRITICAL: Dataset is empty!")
        return
    
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    # Subset wrappers to apply different transforms
    class SubsetWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label
        def __len__(self):
            return len(self.subset)
    
    train_data = SubsetWrapper(train_subset, train_transform)
    val_data = SubsetWrapper(val_subset, val_transform)
    
    # Weighted sampler for training
    train_indices = train_subset.indices
    train_labels = full_dataset.labels[train_indices]
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    print(f"Training Class Counts: {class_counts}")
    
    class_weights_arr = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights_arr[train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model: ResNet-50 (pretrained)
    print(f"Initializing ResNet-50 with {num_classes} output classes...")
    model = models.resnet50(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes),
    )
    model = model.to(DEVICE)
    
    # Loss: Cross Entropy without Label Smoothing (anti-overfitting via Dropout)
    loss_weight = torch.tensor(class_weights_arr / class_weights_arr.sum() * num_classes, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    
    # Optimizer + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    GRAD_ACCUM_STEPS = 8  # Doubled from 4 to maintain Effective Batch Size = 32 with BATCH_SIZE=4
    print(f"Using Gradient Accumulation: {GRAD_ACCUM_STEPS} steps (Effective batch size = {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        optimizer.zero_grad()
        for step, (inputs, labels) in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
            train_loss += loss.item() * GRAD_ACCUM_STEPS
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Scheduler steps on val_loss below

        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels_list = []
        val_loss = 0
        
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
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Saved Best Model (Val Acc: {val_acc:.2%})")
    
    print("\n" + "=" * 60)
    print(f"Training Complete. Best Val Acc: {best_acc:.2%}")
    print("=" * 60)
    
    # Classification Report
    class_names = full_dataset.class_names if hasattr(full_dataset, 'class_names') else None
    if class_names:
        print("\nFinal Validation Classification Report:")
        print(classification_report(
            val_labels_list, val_preds,
            target_names=class_names,
            zero_division=0
        ))


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Visual Router/Specialist Training")
    parser.add_argument("--mode", type=str, default="router",
                        choices=["router", "rhythm", "structure"],
                        help="Training mode: router, rhythm, or structure")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    args = parser.parse_args()
    
    train_visual(mode=args.mode, epochs=args.epochs)
