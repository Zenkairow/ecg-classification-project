import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm

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
LEARNING_RATE = 1e-4 # Low LR for fine-tuning
GAMMA = 2.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Alpha balancing not strictly requested, but good to have capability
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [Batch, NumClasses] (Logits)
        targets: [Batch] (Class Indices)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# ---------------------------------

# --- Helper: Mixup (Local Copy) ---
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion_focal(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# ----------------------------------

def finetune():
    print("--- Starting SE-ResNet Focal Loss Fine-Tuning ---")
    os.makedirs("models", exist_ok=True)
    
    # 1. Data
    print("Loading Dataset...")
    dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, detected_seq_len=5000)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42) # Consistent split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    num_classes = len(dataset.class_map)
    
    # 2. Model
    print("Loading Pre-trained SE-ResNet-34...")
    model = SEResNet34(num_classes=num_classes, input_channels=NUM_LEADS)
    
    # Load Best Weights
    weights_path = 'models/signal_seresnet34_best.pth'
    if not os.path.exists(weights_path):
        weights_path = 'models/signal_resnet34_70acc.pth' # Fallback
        
    try:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded weights from {weights_path}")
    except Exception as e:
        print(f"CRITICAL: Failed to load weights: {e}")
        return

    model = model.to(DEVICE)
    
    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # FOCAL LOSS
    criterion = FocalLoss(gamma=GAMMA)
    
    best_acc = 0.0
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Focal Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Focal Loss with Mixup
            loss = mixup_criterion_focal(criterion, outputs, targets_a, targets_b, lam)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/seresnet_focal_best.pth")
            print("  -> Saved Best Focal Model")
            
    print(f"Fine-tuning Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    finetune()
