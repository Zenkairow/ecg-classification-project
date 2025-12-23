import torch
import numpy as np
import os
import sys

# Add current directory to path so we can import model
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from .model import SEResNet34

# Constants
SIGNAL_LENGTH = 5000
NUM_LEADS = 12

# Class Mappings (Hardcoded for stability in production)
RHYTHM_CLASSES = sorted([
    'Atrial_Fibrillation', 'SVT', 'AV_Block_1st_Deg', 'AV_Block_2nd_Deg', 
    'AV_Block_3rd_Deg', 'PVC', 'Paced', 'Fascicular_Block', 'NDT', 
    'WPW', 'DIG', 'EL', 'Dig'
])

STRUCTURE_CLASSES = sorted([
    'MI_Anterior', 'MI_Inferior', 'MI_Lateral', 'LBBB', 'RBBB', 
    'Left_Hypertrophy', 'Right_Hypertrophy', 'Ischemia', 
    'NonSpecific_ST', 'IVCD', 'LNGQT'
])

class CardiacPredictor:
    def __init__(self, models_dir="models"):
        """
        Initialize the predictor by loading models from the specified directory.
        models_dir: relative path from the production root or absolute path.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device: {self.device}")
        
        # Determine absolute paths
        # Assuming folder structure:
        # production/
        #   app.py
        #   utils/
        #      backend.py
        #   models/
        
        # If we are running from production/ via app.py, models_dir="models" works
        # If we are running from backend.py directly, we need to adjust
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # production/
        models_path = os.path.join(base_path, models_dir)
        
        self.router_path = os.path.join(models_path, "hierarchy_stage2_router.pth")
        self.rhythm_path = os.path.join(models_path, "hierarchy_stage3_rhythm.pth")
        self.structure_path = os.path.join(models_path, "hierarchy_stage3_structure.pth")
        
        self._load_models()

    def _load_models(self):
        print("Loading Models...")
        
        # Router
        self.router = SEResNet34(num_classes=2, input_channels=NUM_LEADS)
        self.router.load_state_dict(torch.load(self.router_path, map_location=self.device))
        self.router.to(self.device).eval()
        
        # Rhythm
        self.rhythm_net = SEResNet34(num_classes=len(RHYTHM_CLASSES), input_channels=NUM_LEADS)
        self.rhythm_net.load_state_dict(torch.load(self.rhythm_path, map_location=self.device))
        self.rhythm_net.to(self.device).eval()
        
        # Structure
        self.structure_net = SEResNet34(num_classes=len(STRUCTURE_CLASSES), input_channels=NUM_LEADS)
        self.structure_net.load_state_dict(torch.load(self.structure_path, map_location=self.device))
        self.structure_net.to(self.device).eval()
        
        print("Models Loaded Successfully.")

    def preprocess(self, signal):
        """Standardize signal to [1, 12, 5000] tensor"""
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32)
            
        # Ensure specific shape
        # Expected input: [12, L] or [L, 12] or [1, 12, L]
        if signal.dim() == 2:
             # If [L, 12] -> Transpose to [12, L]
             if signal.shape[0] != 12 and signal.shape[1] == 12:
                 signal = signal.T
             signal = signal.unsqueeze(0) # [1, 12, L]
        elif signal.dim() == 3:
             if signal.shape[1] != 12 and signal.shape[2] == 12:
                 signal = signal.permute(0, 2, 1)
             
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
                top3 = {RHYTHM_CLASSES[i]: p.item() for i, p in zip(top3_idx[0], top3_prob[0])}

            else:
                # Structure Path
                path_name = "Structure"
                logits = self.structure_net(input_tensor)
                probs = torch.softmax(logits, dim=1)
                idx = torch.argmax(probs, dim=1).item()
                
                diagnosis = STRUCTURE_CLASSES[idx]
                confidence = probs[0, idx].item()
                
                top3_prob, top3_idx = torch.topk(probs, 3)
                top3 = {STRUCTURE_CLASSES[i]: p.item() for i, p in zip(top3_idx[0], top3_prob[0])}
            
            return {
                "triage": path_name,
                "diagnosis": diagnosis,
                "confidence": confidence,
                "top3": top3, # Dictionary for chart
                "router_conf": router_prob[0, router_pred].item()
            }
