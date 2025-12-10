import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from models.transformer import ECGTransformer

# --- Configuration ---
BATCH_SIZE = 16  # Reduced BS for larger models/sequences
NUM_LEADS = 12
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50 
DATA_DIR = 'data_synthesis/output/output/signals/'
CSV_PATH = 'data/train_labels.csv'
MODEL_SAVE_PATH = 'models/signal_transformer_v6_best.pth'

# --- Clinical Taxonomy ---
def group_diagnostic_classes(label):
    """
    Reduces 50 complex classes into ~20 major clinical categories.
    Consistency with Engine A.
    """
    label = str(label)
    # 1. Myocardial Infarction (MI)
    if label in ['AMI', 'ALMI', 'ASMI', 'INJAL', 'INJAS']: return 'MI_Anterior'
    if label in ['IMI', 'ILMI', 'IPLMI', 'IPMI', 'INJIL', 'INJIN']: return 'MI_Inferior'
    if label in ['LMI', 'INJLA', 'PMI']: return 'MI_Lateral'
    
    # 2. Ischemia & ST Changes (Split)
    if 'ISC' in label: return 'Ischemia'
    if label == 'NST_': return 'NonSpecific_ST'
    
    # 3. Bundle Branch Blocks
    if label in ['CLBBB', 'ILBBB']: return 'LBBB'
    if label in ['CRBBB', 'IRBBB']: return 'RBBB'
    if label == 'IVCD': return 'IVCD'
    
    # 4. AV Blocks (Split by severity)
    if label == '1AVB': return 'AV_Block_1st_Deg'
    if label == '2AVB': return 'AV_Block_2nd_Deg'
    if label == '3AVB': return 'AV_Block_3rd_Deg'
    
    # 5. Hypertrophy
    if label in ['LVH', 'LAO/LAE']: return 'Left_Hypertrophy'
    if label in ['RVH', 'RAO/RAE', 'SEHYP']: return 'Right_Hypertrophy'
    
    # 6. Fascicular Blocks
    if label in ['LAFB', 'LPFB']: return 'Fascicular_Block'
    
    # 7. Rhythms
    if label in ['AFIB', 'AFLT']: return 'Atrial_Fibrillation'
    if label in ['SARRH', 'STACH', 'SBRAD', 'SR']: return 'Sinus_Rhythm'
    if label == 'PACE': return 'Paced'
    if label in ['PSVT', 'SVT']: return 'SVT'
    
    # 8. Others
    if label == 'NORM': return 'NORM'
    
    return label

# --- Dataset Class ---
class ECGSignalDataset(Dataset):
    def __init__(self, csv_file, root_dir, detected_seq_len=None):
        self.root_dir = root_dir
        self.seq_len = detected_seq_len
        
        # Load CSV
        try:
            self.annotations = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            sys.exit(1)
            
        # Class Mapping (Grouped)
        loss_col = 'label' if 'label' in self.annotations.columns else 'diagnostic_superclass'
        
        # Gather all grouped labels
        unique_groups = set()
        for raw in self.annotations[loss_col].unique():
            unique_groups.add(group_diagnostic_classes(raw))
            
        self.unique_labels = sorted(list(unique_groups))
        self.class_map = {label: i for i, label in enumerate(self.unique_labels)}
        
        print(f"Signal Dataset Loaded. Total samples: {len(self.annotations)}")
        print(f"Clinical Taxonomy Applied: {len(self.unique_labels)} Classes.")
        print(f"Classes: {self.unique_labels}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 1. Get Filename 
        row = self.annotations.iloc[index]
        base_name = str(row['filename'])
        if base_name.endswith('.png'):
            base_name = base_name.replace('.png', '.npy')
        elif not base_name.endswith('.npy'):
            base_name = base_name + '.npy'
            
        signal_path = os.path.join(self.root_dir, base_name)
        
        # 2. Get Label (Grouped)
        loss_col = 'label' if 'label' in self.annotations.columns else 'diagnostic_superclass'
        raw_label = str(row[loss_col])
        grouped = group_diagnostic_classes(raw_label)
        label_id = self.class_map[grouped]
        
        # 3. Load Signal
        signal_tensor = torch.zeros((NUM_LEADS, self.seq_len)).float() # Default
        
        try:
            # Load Signal
            signal = np.load(signal_path) # [12, L] or [L, 12]
            
            # Align Dimensions -> [12, L]
            if signal.shape[0] != NUM_LEADS and signal.shape[1] == NUM_LEADS:
                signal = signal.T 

            # --- RESAMPLING / STANDARDIZATION ---
            # We enforce consistency. If signal != self.seq_len, we resample.
            # This handles mixed 100Hz/500Hz datasets correctly.
            current_len = signal.shape[1]
            
            if current_len != self.seq_len:
                # Use linear interpolation for efficiency (and PyTorch compatibility later)
                # x_old = np.linspace(0, 1, current_len)
                # x_new = np.linspace(0, 1, self.seq_len)
                # target_signal = np.zeros((NUM_LEADS, self.seq_len))
                # for lead in range(NUM_LEADS):
                #     target_signal[lead] = np.interp(x_new, x_old, signal[lead])
                # signal = target_signal
                
                # Faster approach using scipy or simple expansion
                # Here we use a simple resize logic via torch interpolation (GPU ready logic, but done on CPU here)
                sig_t = torch.tensor(signal).unsqueeze(0) # [1, 12, L]
                sig_t = torch.nn.functional.interpolate(sig_t, size=self.seq_len, mode='linear', align_corners=False)
                signal = sig_t.squeeze(0).numpy()
            
            # Normalize (Z-score)
            mean = np.mean(signal, axis=1, keepdims=True)
            std = np.std(signal, axis=1, keepdims=True)
            std[std == 0] = 1.0 # Protect division
            signal = (signal - mean) / std
            
            signal_tensor = torch.from_numpy(signal).float()
            
        except Exception as e:
            # print(f"Error loading {signal_path}: {e}")
            pass
            
        return signal_tensor, label_id

def detect_sequence_length(root_dir, csv_path):
    """
    Peeks at the first valid .npy file to determine SEQ_LEN (1000 vs 5000).
    """
    print("Detecting Signal Sequence Length...")
    df = pd.read_csv(csv_path)
    
    for i in range(min(50, len(df))): # Try first 50 entries
        fname = str(df.iloc[i]['filename']).replace('.png', '.npy')
        fpath = os.path.join(root_dir, fname)
        if os.path.exists(fpath):
            try:
                sig = np.load(fpath)
                # Find the larger dimension that isn't 12 (or if 12, assume the other is length)
                shape = sig.shape
                length = max(shape) if min(shape) == 12 else shape[0] # Fallback logic
                print(f"Detected Shape: {shape} -> Using SEQ_LEN = {length}")
                return length
            except:
                continue
    
    print("Warning: Could not detect length. Defaulting to 1000.")
    return 1000

# --- Training Function ---
def train_model():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Detect Length
    seq_len = detect_sequence_length(DATA_DIR, CSV_PATH)
    print(f"Training Config: SEQ_LEN={seq_len}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={NUM_EPOCHS}")

    # 2. Dataset
    dataset = ECGSignalDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, detected_seq_len=seq_len)
    
    if len(dataset) == 0:
        print("CRITICAL: Dataset is empty!")
        return

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 3. Model
    print("Initializing ECGTransformer (v6 High-Spec)...")
    num_classes = len(dataset.class_map)
    
    # High Quality settings: d_model=512 (if seq_len is huge, we might need to reduce batch size)
    d_model = 512
    if seq_len > 2000:
        d_model = 256 # Save memory for very long sequences
        
    model = ECGTransformer(
        num_classes=num_classes, 
        input_channels=NUM_LEADS,
        seq_len=seq_len,
        d_model=d_model, 
        nhead=8
    )
    model = model.to(device)

    # 4. Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 5. Training Loop
    best_val_acc = 0.0
    print(f"Starting Signal Training...")
    
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
