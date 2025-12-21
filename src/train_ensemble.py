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

# Setup Path to import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

# Imports from existing codebase
from src.engine_b_signal.train_signal_model import ECGSignalDataset, CSV_PATH, DATA_DIR, mixup_data, mixup_criterion, NUM_LEADS
from src.engine_b_signal.models.resnet1d import ResNet1D
from src.engine_b_signal.models.resnet1d_se import SEResNet34

# Configuration
BATCH_SIZE = 32
EPOCHS = 15 # Fast training for specialists
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specialist Definitions
# Leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
# Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
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

def get_dataloaders():
    """Consistent split with training/diagnosis"""
    dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, detected_seq_len=5000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42) # Deterministic split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, dataset.class_map

def train_specialist(name, config, train_loader, val_loader, num_classes):
    print(f"\nTraining Specialist: {name} ({config['description']})")
    
    # Lightweight ResNet-18 (layers=[2,2,2,2])
    model = ResNet1D(num_classes=num_classes, input_channels=config['channels'], layers=[2, 2, 2, 2])
    model = model.to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # Slightly different LR for smaller model? Keep standard.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    indices = torch.tensor(config['indices']).to(DEVICE)
    
    best_acc = 0.0
    save_path = f"models/specialist_{name.lower()}.pth"
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Training Loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # SLICE LEADS
            inputs = inputs[:, indices, :]
            
            # Mixup
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

def run_ensemble(global_model_path, specialist_paths, val_loader, class_map, num_classes):
    print("\n" + "="*50)
    print("RUNNING GRAND UNIFIED ENSEMBLE")
    print("="*50)
    
    # 1. Load Models
    models = {}
    
    # Global (SE-ResNet-34)
    print("Loading Global Model...")
    global_model = SEResNet34(num_classes=num_classes, input_channels=12) # Full 12 leads
    try:
        ckpt = torch.load(global_model_path, map_location=DEVICE)
        
        # Handle 'model_state_dict' key if present
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
            
        # Clean state dict keys if needed (e.g. remove 'module.')
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        global_model.load_state_dict(clean_state_dict, strict=False) # strict=False to be safe with partial matches if any
        global_model.to(DEVICE)
        global_model.eval()
        models['Global'] = global_model
    except Exception as e:
        print(f"CRITICAL: Failed to load global model: {e}")
        return

    # Specialists
    for name, path in specialist_paths.items():
        print(f"Loading {name} Specialist...")
        spec_model = ResNet1D(num_classes=num_classes, input_channels=SPECIALISTS[name]['channels'], layers=[2,2,2,2])
        spec_model.load_state_dict(torch.load(path, map_location=DEVICE))
        spec_model.to(DEVICE)
        spec_model.eval()
        models[name] = spec_model

    # 2. Ensemble Inference
    all_preds = []
    all_labels = []
    
    # Weights
    W_GLOBAL = 0.4
    W_SPEC = 0.2
    
    # Indices tensors for slicing
    idx_ant = torch.tensor(SPECIALISTS['Anterior']['indices']).to(DEVICE)
    idx_inf = torch.tensor(SPECIALISTS['Inferior']['indices']).to(DEVICE)
    idx_lat = torch.tensor(SPECIALISTS['Lateral']['indices']).to(DEVICE)
    
    print("Evaluating Ensemble...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Ensemble Inference"):
            inputs = inputs.to(DEVICE)
            
            # Forward Passes
            # Global (Full 12)
            out_global = torch.softmax(models['Global'](inputs), dim=1)
            
            # Specialists (Sliced)
            out_ant = torch.softmax(models['Anterior'](inputs[:, idx_ant, :]), dim=1)
            out_inf = torch.softmax(models['Inferior'](inputs[:, idx_inf, :]), dim=1)
            out_lat = torch.softmax(models['Lateral'](inputs[:, idx_lat, :]), dim=1)
            
            # Weighted Sum
            final_prob = (W_GLOBAL * out_global) + \
                         (W_SPEC * out_ant) + \
                         (W_SPEC * out_inf) + \
                         (W_SPEC * out_lat)
            
            _, predicted = torch.max(final_prob, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 3. Report
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nENSEMBLE ACCURACY: {acc:.2%}")
    
    target_names = [k for k,v in sorted(class_map.items(), key=lambda item: item[1])]
    print(classification_report(all_labels, all_preds, target_names=target_names, labels=range(len(class_map))))

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    train_loader, val_loader, class_map = get_dataloaders()
    num_classes = len(class_map)
    
    # 1. Train Specialists
    specialist_paths = {}
    for name, config in SPECIALISTS.items():
        path = train_specialist(name, config, train_loader, val_loader, num_classes)
        specialist_paths[name] = path
        
    # 2. Run Ensemble
    # Try to find best global model
    global_model = 'models/signal_seresnet34_best.pth'
    if not os.path.exists(global_model):
        global_model = 'models/signal_resnet34_70acc.pth'
        
    run_ensemble(global_model, specialist_paths, val_loader, class_map, num_classes)
