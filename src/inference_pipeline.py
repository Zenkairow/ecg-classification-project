import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import os
import random

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

# Imports
from src.config import (
    STAGE2_MODEL_PATH, 
    STAGE3_RHYTHM_MODEL_PATH, 
    STAGE3_STRUCTURE_MODEL_PATH, 
    SIGNAL_LENGTH
)
from src.engine_b_signal.train_signal_model import DATA_DIR, NUM_LEADS, CSV_PATH, group_diagnostic_classes
from src.engine_b_signal.models.resnet1d_se import SEResNet34
from src.prepare_stage2_data import CLASS_0_RHYTHM, CLASS_1_STRUCTURE

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Classes Mapping (Must match Training exactly) ---
RHYTHM_CLASSES = sorted(CLASS_0_RHYTHM)
STRUCTURE_CLASSES = sorted(CLASS_1_STRUCTURE)

class CardiacSystem:
    def __init__(self):
        print("--- Initializing Cardiac Diagnostic System ---")
        self.device = DEVICE
        
        # 1. Load Router
        print(f"Loading Router: {STAGE2_MODEL_PATH}")
        self.router = SEResNet34(num_classes=2, input_channels=NUM_LEADS)
        self.router.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=self.device))
        self.router.to(self.device).eval()
        
        # 2. Load Rhythm Specialist
        print(f"Loading Rhythm Specialist: {STAGE3_RHYTHM_MODEL_PATH}")
        self.rhythm_net = SEResNet34(num_classes=len(RHYTHM_CLASSES), input_channels=NUM_LEADS)
        self.rhythm_net.load_state_dict(torch.load(STAGE3_RHYTHM_MODEL_PATH, map_location=self.device))
        self.rhythm_net.to(self.device).eval()
        
        # 3. Load Structure Specialist
        print(f"Loading Structure Specialist: {STAGE3_STRUCTURE_MODEL_PATH}")
        self.structure_net = SEResNet34(num_classes=len(STRUCTURE_CLASSES), input_channels=NUM_LEADS)
        self.structure_net.load_state_dict(torch.load(STAGE3_STRUCTURE_MODEL_PATH, map_location=self.device))
        self.structure_net.to(self.device).eval()
        
        print(">>> System Ready.")

    def preprocess(self, signal):
        """Standardize signal to [1, 12, 5000]"""
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32)
            
        # Ensure specific shape
        if signal.dim() == 2:
             # [12, L] or [L, 12]
             if signal.shape[0] != 12 and signal.shape[1] == 12:
                 signal = signal.T
             signal = signal.unsqueeze(0) # [1, 12, L]
             
        # Interpolate if needed
        if signal.shape[2] != SIGNAL_LENGTH:
            signal = torch.nn.functional.interpolate(signal, size=SIGNAL_LENGTH, mode='linear')
            
        return signal.to(self.device)

    def predict(self, signal_numpy):
        """
        Hierarchical Inference Strategy:
        1. Router determines Path: Rhythm (0) vs Structure (1)
        2. Specialist determines exact Diagnosis
        """
        input_tensor = self.preprocess(signal_numpy)
        
        with torch.no_grad():
            # --- Step 1: Router ---
            router_logits = self.router(input_tensor)
            router_prob = torch.softmax(router_logits, dim=1)
            router_pred = torch.argmax(router_prob, dim=1).item()
            
            # --- Step 2: Specialist ---
            result = {}
            if router_pred == 0:
                # Rhythm Path
                path_name = "Rhythm"
                logits = self.rhythm_net(input_tensor)
                probs = torch.softmax(logits, dim=1)
                idx = torch.argmax(probs, dim=1).item()
                
                diagnosis = RHYTHM_CLASSES[idx]
                confidence = probs[0, idx].item()
                
                # Top 3
                top3_prob, top3_idx = torch.topk(probs, 3)
                top3 = [(RHYTHM_CLASSES[i], p.item()) for i, p in zip(top3_idx[0], top3_prob[0])]

            else:
                # Structure Path
                path_name = "Structure"
                logits = self.structure_net(input_tensor)
                probs = torch.softmax(logits, dim=1)
                idx = torch.argmax(probs, dim=1).item()
                
                diagnosis = STRUCTURE_CLASSES[idx]
                confidence = probs[0, idx].item()
                
                top3_prob, top3_idx = torch.topk(probs, 3)
                top3 = [(STRUCTURE_CLASSES[i], p.item()) for i, p in zip(top3_idx[0], top3_prob[0])]
            
            return {
                "triage": path_name,
                "diagnosis": diagnosis,
                "confidence": confidence,
                "top3": top3,
                "router_conf": router_prob[0, router_pred].item()
            }

def run_demo():
    print("\n\n--- Running Demo on Random Samples ---")
    
    # Init System
    system = CardiacSystem()
    
    # Load Master CSV
    df = pd.read_csv(CSV_PATH)
    
    # Pick 5 random Abnormal samples to test (skipping Normals for now as requested)
    # Normals are filtered out because trained specialists don't know "Normal" class
    # and would hallucinate a disease.
    # Safe filtering
    label_mask = ~df['label'].isin(['NORM', 'Sinus_Rhythm'])
    if 'diagnostic_superclass' in df.columns:
        superclass_mask = ~df['diagnostic_superclass'].isin(['NORM', 'Sinus_Rhythm'])
        valid_mask = label_mask & superclass_mask
    else:
        valid_mask = label_mask
        
    sub_df = df[valid_mask]
    
    samples = sub_df.sample(5)
    
    for idx, row in samples.iterrows():
        print(f"\n------------------------------------------------")
        
        # Load File
        fname = row.get('filename_hr')
        if pd.isna(fname): fname = row.get('filename') # Fallback
        if str(fname).endswith('.png'): fname = str(fname).replace('.png', '.npy')
        if not str(fname).endswith('.npy'): fname = str(fname) + '.npy'
            
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Skipping {fname} (Not Found)")
            continue
            
        signal = np.load(fpath)
        
        # Ground Truth
        raw_label = str(row.get('label', 'Unknown'))
        group_label = group_diagnostic_classes(raw_label)
        
        print(f"File: {fname}")
        print(f"Ground Truth: {raw_label} -> {group_label}")
        
        # Predict
        pred = system.predict(signal)
        
        print(f"System Prediction:")
        print(f"  > Triage:    {pred['triage']} (Conf: {pred['router_conf']:.1%})")
        print(f"  > Diagnosis: {pred['diagnosis']} (Conf: {pred['confidence']:.1%})")
        
        # Check correctness
        is_correct_triage = False
        if pred['triage'] == "Rhythm" and group_label in CLASS_0_RHYTHM: is_correct_triage = True
        if pred['triage'] == "Structure" and group_label in CLASS_1_STRUCTURE: is_correct_triage = True
        
        is_correct_diag = (pred['diagnosis'] == group_label)
        
        triage_mark = "✅" if is_correct_triage else "❌"
        diag_mark = "✅" if is_correct_diag else "❌"
        
        print(f"Result: Triage {triage_mark} | Diagnosis {diag_mark}")

if __name__ == "__main__":
    run_demo()
