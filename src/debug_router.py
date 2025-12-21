import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sys
import os

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

# Imports
from src.config import STAGE2_DATA_PATH
from src.engine_b_signal.train_signal_model import DATA_DIR, NUM_LEADS
from src.engine_b_signal.models.resnet1d_se import SEResNet34

# Mock Dataset Logic (Copy-Paste from script to reproduce exact behavior)
class DebugRouterDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, seq_len=5000):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        fname = row.get('filename_hr', row.get('filename_lr'))
        if not isinstance(fname, str):
             fname = str(fname)
        
        file_path = os.path.join(self.root_dir, fname)
        if not file_path.endswith('.npy'):
             file_path += '.npy'
             
        status = "OK"
        try:
            signal = np.load(file_path)
            if np.all(signal == 0):
                status = "ZERO_CONTENT"
        except Exception as e:
            status = f"MISSING/ERROR: {e}"
            signal = np.zeros((12, self.seq_len))
            
        print(f"Debug Item {idx}: Path={file_path} | Status={status} | Shape={signal.shape}")
        
        # Processing logic
        if signal.shape[0] != 12 and signal.shape[1] == 12:
            signal = signal.T
            
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(int(row['router_label']), dtype=torch.long)

def run_debug():
    print("--- Debugging Stage 2 Data Loading ---")
    print(f"Reading {STAGE2_DATA_PATH}")
    
    if not os.path.exists(STAGE2_DATA_PATH):
        print("CSV Missing!")
        return
        
    ds = DebugRouterDataset(STAGE2_DATA_PATH, DATA_DIR)
    
    # Check first 5 items
    for i in range(5):
        ds[i]
        
    # Check Model
    print("\n--- Checking Model Forward Pass ---")
    model = SEResNet34(num_classes=2, input_channels=NUM_LEADS)
    dummy_input = torch.randn(2, 12, 5000)
    out = model(dummy_input)
    print(f"Forward Pass Output Shape: {out.shape}")
    print(f"Logits: {out.detach().numpy()}")
    print("Model check passed if logits are not NaN.")

if __name__ == "__main__":
    run_debug()
