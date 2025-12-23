import torch
import torch.nn as nn
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
from src.engine_b_signal.train_signal_model import ECGSignalDataset, CSV_PATH, DATA_DIR
from src.engine_b_signal.models.resnet1d import ResNet1D

# Configuration
BATCH_SIZE = 32
EPOCHS_WARMUP = 5
EPOCHS_FINETUNE = 10
TOTAL_EPOCHS = EPOCHS_WARMUP + EPOCHS_FINETUNE
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hydra Attention Network ---
class HydraAttentionNet(nn.Module):
    def __init__(self, num_classes):
        super(HydraAttentionNet, self).__init__()
        
        # 1. Backbones (Specialists)
        self.branch_ant = ResNet1D(num_classes=num_classes, input_channels=4, layers=[2,2,2,2])
        self.branch_inf = ResNet1D(num_classes=num_classes, input_channels=3, layers=[2,2,2,2])
        self.branch_lat = ResNet1D(num_classes=num_classes, input_channels=4, layers=[2,2,2,2])
        
        # Lead Indices
        self.idx_ant = [6, 7, 8, 9] # V1-V4
        self.idx_inf = [1, 2, 5]    # II, III, aVF
        self.idx_lat = [0, 4, 10, 11] # I, aVL, V5, V6
        
        # Feature Dimensions
        # ResNet18 (BasicBlock) * expansion(1) * 512 channels at end = 512
        self.feat_dim = 512
        
        # 2. Attention Gate
        # Input: All features concatenated (512 * 3 = 1536)
        self.attention_gate = nn.Sequential(
            nn.Linear(self.feat_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 3), # Output: 3 weights [a1, a2, a3]
            nn.Softmax(dim=1)
        )
        
        # 3. Classifier
        # Input: Weighted features still have size 1536 (just scaled)
        # OR: We could sum them if they shared the same latent space, but they are specialist spaces.
        # User specified: "Concatenate the weighted vectors."
        self.classifier = nn.Linear(self.feat_dim * 3, num_classes)
        
    def load_specialists_pure(self):
        """Loads weights from pure training run."""
        print("Loading Pure Specialist Weights...")
        paths = {
            'Anterior': 'models/specialist_anterior_pure.pth',
            'Inferior': 'models/specialist_inferior_pure.pth',
            'Lateral':  'models/specialist_lateral_pure.pth'
        }
        branches = [self.branch_ant, self.branch_inf, self.branch_lat]
        
        for branch, name in zip(branches, paths.keys()):
            path = paths[name]
            if os.path.exists(path):
                print(f"  - Loading {name} from {path}")
                try:
                    state_dict = torch.load(path, map_location=DEVICE)
                    branch.load_state_dict(state_dict)
                except Exception as e:
                    print(f"    WARNING: Failed to load {name}: {e}")
            else:
                print(f"  - WARNING: {path} not found. Random init.")
                
            # Remove Head (FC) to make them Feature Extractors
            branch.fc = nn.Identity()

    def set_backbones_trainable(self, trainable=True):
        """Freeze or Unfreeze backbones."""
        branches = [self.branch_ant, self.branch_inf, self.branch_lat]
        for branch in branches:
            for param in branch.parameters():
                param.requires_grad = trainable
                
    def forward(self, x):
        # 1. Slice
        x_ant = x[:, self.idx_ant, :]
        x_inf = x[:, self.idx_inf, :]
        x_lat = x[:, self.idx_lat, :]
        
        # 2. Extract Features (512 each)
        f_ant = self.branch_ant(x_ant)
        f_inf = self.branch_inf(x_inf)
        f_lat = self.branch_lat(x_lat)
        
        # 3. Attention Calculation
        # Stack to form context vector: [Batch, 1536]
        concat_features = torch.cat([f_ant, f_inf, f_lat], dim=1)
        
        # Calculate attention weights: [Batch, 3]
        attn_weights = self.attention_gate(concat_features)
        
        # 4. Weighted Fusion
        # Expand weights for multiplication: [Batch, 1]
        w_ant = attn_weights[:, 0].unsqueeze(1)
        w_inf = attn_weights[:, 1].unsqueeze(1)
        w_lat = attn_weights[:, 2].unsqueeze(1)
        
        # Scale each feature vector
        f_ant_w = f_ant * w_ant
        f_inf_w = f_inf * w_inf
        f_lat_w = f_lat * w_lat
        
        # Concat again for classification
        f_final = torch.cat([f_ant_w, f_inf_w, f_lat_w], dim=1)
        
        # 5. Classify
        out = self.classifier(f_final)
        return out

def train():
    print("--- Starting Hydra 2.0 Phase 2: Attention Fusion ---")
    os.makedirs("models", exist_ok=True)
    
    # Data
    dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, detected_seq_len=5000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    num_classes = len(dataset.class_map)
    
    # Model
    model = HydraAttentionNet(num_classes)
    model.load_specialists_pure()
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    # --- TRAINING LOOP ---
    for epoch in range(TOTAL_EPOCHS):
        
        # PHASE SHIFT LOGIC
        if epoch < EPOCHS_WARMUP:
            phase = "PHASE A (WARMUP - HEAD ONLY)"
            model.set_backbones_trainable(False)
            current_lr = LEARNING_RATE
        else:
            phase = "PHASE B (FINETUNE - FULL MODEL)"
            model.set_backbones_trainable(True)
            current_lr = LEARNING_RATE / 10 # 1e-4
            
        print(f"\nEpoch {epoch+1}/{TOTAL_EPOCHS} - {phase} - LR: {current_lr}")
        
        # Re-init optimizer if phase changes (or just update param groups)
        # Re-init is safer to catch graph changes or require_grad changes.
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr)
        
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
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
        print(f" Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/hydra_attention_best.pth")
            print("  -> Saved Best Hydra Attention Model")

if __name__ == "__main__":
    train()
