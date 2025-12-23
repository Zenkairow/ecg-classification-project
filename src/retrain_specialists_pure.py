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
from src.engine_b_signal.train_signal_model import ECGSignalDataset, CSV_PATH, DATA_DIR, NUM_LEADS
from src.engine_b_signal.models.resnet1d import ResNet1D

# Configuration
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specialist Definitions
SPECIALISTS = {
    'Anterior': {
        'indices': [6, 7, 8, 9], # V1-V4
        'channels': 4,
        'description': 'Focus: Anterior/Septal MI, RBBB'
    },
    'Inferior': {
        'indices': [1, 2, 5], # II, III, aVF
        'channels': 3,
        'description': 'Focus: Inferior MI'
    },
    'Lateral': {
        'indices': [0, 4, 10, 11], # I, aVL, V5, V6
        'channels': 4,
        'description': 'Focus: Lateral MI, LBBB, LVH'
    }
}

def train_specialist(name, config, train_loader, val_loader, num_classes):
    print(f"\n--- Training Specialist: {name} (Pure Data) ---")
    print(f"Leads: {config['channels']} channels")
    
    # Lightweight ResNet-18
    model = ResNet1D(num_classes=num_classes, input_channels=config['channels'], layers=[2, 2, 2, 2])
    model = model.to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # CRITICAL: Standard CrossEntropy (No Label Smoothing)
    criterion = nn.CrossEntropyLoss()
    
    indices = torch.tensor(config['indices']).to(DEVICE)
    
    best_acc = 0.0
    save_path = f"models/specialist_{name.lower()}_pure.pth"
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"{name} Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # SLICE LEADS
            inputs = inputs[:, indices, :]
            
            # NO MIXUP - Pure Training
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
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
                inputs = inputs[:, indices, :] # Slice
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            
    print(f"Best {name} Accuracy: {best_acc:.2f}% | Saved to {save_path}")
    return save_path

if __name__ == "__main__":
    print("--- Starting Hydra 2.0 Phase 1: Pure Specialist Training ---")
    os.makedirs("models", exist_ok=True)
    
    # Load Data
    print("Loading Dataset...")
    dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, detected_seq_len=5000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    num_classes = len(dataset.class_map)
    
    # Train Loop
    for name, config in SPECIALISTS.items():
        train_specialist(name, config, train_loader, val_loader, num_classes)
