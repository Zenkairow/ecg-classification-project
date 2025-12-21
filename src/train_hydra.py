import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

# Imports
from src.engine_b_signal.train_signal_model import ECGSignalDataset, CSV_PATH, DATA_DIR, NUM_LEADS
from src.engine_b_signal.models.resnet1d import ResNet1D

# Config
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4 # Low LR for fine-tuning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Inline Mixup ---
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

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# --------------------

class HydraNet(nn.Module):
    def __init__(self, num_classes):
        super(HydraNet, self).__init__()
        
        # Branch Definitions
        # ResNet1D(layers=[2,2,2,2]) = ResNet-18 Light
        self.branch_ant = ResNet1D(num_classes=num_classes, input_channels=4, layers=[2,2,2,2])
        self.branch_inf = ResNet1D(num_classes=num_classes, input_channels=3, layers=[2,2,2,2])
        self.branch_lat = ResNet1D(num_classes=num_classes, input_channels=4, layers=[2,2,2,2])
        
        # Lead Indices
        self.idx_ant = [6, 7, 8, 9] # V1-V4
        self.idx_inf = [1, 2, 5]    # II, III, aVF
        self.idx_lat = [0, 4, 10, 11] # I, aVL, V5, V6
        
        # Fusion Layer
        # Each ResNet1D output is 512 (if we take the output before FC, need to verify)
        # ResNet1D.fc is Linear(512 * expansion, num_classes). expansion=1 for BasicBlock.
        # So feature dim is 512.
        self.fusion = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def load_specialists(self):
        """Loads weights into branches and then removes their FC layers"""
        print("Loading Specialist Weights (Warm Start)...")
        
        paths = {
            'Anterior': 'models/specialist_anterior.pth',
            'Inferior': 'models/specialist_inferior.pth',
            'Lateral':  'models/specialist_lateral.pth'
        }
        
        branches = [self.branch_ant, self.branch_inf, self.branch_lat]
        names = ['Anterior', 'Inferior', 'Lateral']
        
        for branch, name in zip(branches, names):
            path = paths[name]
            if os.path.exists(path):
                print(f"  - Loading {name} from {path}")
                try:
                    state_dict = torch.load(path, map_location=DEVICE)
                    branch.load_state_dict(state_dict)
                except Exception as e:
                    print(f"    WARNING: Failed to load {name}: {e}")
            else:
                print(f"  - WARNING: {path} not found. Initializing randomly.")
                
            # Replacing FC with Identity to turn them into Feature Extractors
            branch.fc = nn.Identity()

    def forward(self, x):
        # x: [Batch, 12, 5000]
        
        # Slice inputs
        x_ant = x[:, self.idx_ant, :]
        x_inf = x[:, self.idx_inf, :]
        x_lat = x[:, self.idx_lat, :]
        
        # Extract Features (Result is [Batch, 512] because fc is Identity)
        f_ant = self.branch_ant(x_ant)
        f_inf = self.branch_inf(x_inf)
        f_lat = self.branch_lat(x_lat)
        
        # Concatenate: [Batch, 1536]
        f_cat = torch.cat([f_ant, f_inf, f_lat], dim=1)
        
        # Fusion
        out = self.fusion(f_cat)
        
        return out

def train_hydra():
    print("--- Starting HydraNet Training (Fresh Initialization) ---")
    os.makedirs("models", exist_ok=True)
    
    # Data
    dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, detected_seq_len=5000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    num_classes = len(dataset.class_map)
    print(f"Classes: {num_classes}")
    
    # Model
    model = HydraNet(num_classes=num_classes)
    model.load_specialists()
    model = model.to(DEVICE)
    
    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Hydra Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/hydra_fusion_best.pth")
            print("  -> Saved Best Hydra Model")
            
    print(f"Training Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_hydra()
