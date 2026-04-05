import torch
import torch.nn as nn
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
        # If we are running from backend.py directly, we        # Paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_path = os.path.join(project_root, models_dir)
        
        self.router_path = os.path.join(models_path, "prod_signal_router_v6.pth")
        self.rhythm_path = os.path.join(models_path, "prod_signal_rhythm_v6.pth")
        self.structure_path = os.path.join(models_path, "prod_signal_structure_v6.pth")
        
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

    # ── MC-Dropout Utilities ──────────────────────────────────────
    @staticmethod
    def _enable_mc_dropout(model):
        """
        Force Dropout layers to remain active during inference.
        model.eval() disables dropout — we selectively re-enable it.
        """
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @staticmethod
    def _disable_mc_dropout(model):
        """Restore all Dropout layers to eval mode."""
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

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

    def predict(self, signal_numpy, uncertainty_mode=False, mc_samples=50, alpha=0.1):
        """
        Hierarchical Inference Strategy:
        1. Router determines Path: Rhythm (0) vs Structure (1)
        2. Specialist determines exact Diagnosis
        
        Args:
            signal_numpy: Raw ECG signal [12, L] or [L, 12].
            uncertainty_mode: If True, enable MC-Dropout Conformal Prediction.
            mc_samples: Number of stochastic forward passes (T). Default: 50.
            alpha: Significance level for conformal prediction set. Default: 0.1 (90% coverage).
            
        Returns:
            dict with diagnosis, confidence, top3, and (if uncertainty_mode)
            prediction_set, mean_probs, variance, epistemic_uncertainty.
        """
        input_tensor = self.preprocess(signal_numpy)
        
        with torch.no_grad():
            # --- Step 1: Router (deterministic — always eval mode) ---
            router_logits = self.router(input_tensor)
            router_prob = torch.softmax(router_logits, dim=1)
            router_pred = torch.argmax(router_prob, dim=1).item()
            
            # --- Step 2: Specialist ---
            if router_pred == 0:
                path_name = "Rhythm"
                specialist = self.rhythm_net
                class_names = RHYTHM_CLASSES
            else:
                path_name = "Structure"
                specialist = self.structure_net
                class_names = STRUCTURE_CLASSES
            
            if uncertainty_mode:
                # ─── MC-Dropout Conformal Prediction ─────────────────
                self._enable_mc_dropout(specialist)
                
                # Collect T stochastic forward passes
                mc_probs = []
                for _ in range(mc_samples):
                    logits = specialist(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    mc_probs.append(probs.cpu().numpy()[0])  # [C]
                
                self._disable_mc_dropout(specialist)
                
                # Stack: [T, C]
                mc_probs = np.array(mc_probs)
                
                # Mean probability per class (predictive posterior)
                mean_probs = mc_probs.mean(axis=0)  # [C]
                
                # Variance per class (epistemic uncertainty proxy)
                var_probs = mc_probs.var(axis=0)     # [C]
                
                # Scalar epistemic uncertainty = mean of per-class variance
                epistemic_uncertainty = float(var_probs.mean())
                
                # Top-1 diagnosis from mean probabilities
                top1_idx = int(np.argmax(mean_probs))
                diagnosis = class_names[top1_idx]
                confidence = float(mean_probs[top1_idx])
                
                # ─── Conformal Prediction Set ────────────────────────
                # Nonconformity score: 1 - mean_prob (lower score = more conforming)
                # Include all classes whose mean_prob >= threshold
                # Threshold = alpha (classes with >= alpha probability are "plausible")
                threshold = alpha
                prediction_set = []
                for i, p in enumerate(mean_probs):
                    if p >= threshold:
                        prediction_set.append({
                            "class": class_names[i],
                            "mean_prob": float(p),
                            "variance": float(var_probs[i]),
                            "std": float(np.sqrt(var_probs[i])),
                        })
                
                # Sort prediction set by mean_prob descending
                prediction_set.sort(key=lambda x: x["mean_prob"], reverse=True)
                
                # Top 3 from mean probabilities
                top3_indices = np.argsort(mean_probs)[::-1][:3]
                top3 = {class_names[i]: float(mean_probs[i]) for i in top3_indices}
                
                return {
                    "triage": path_name,
                    "diagnosis": diagnosis,
                    "confidence": confidence,
                    "top3": top3,
                    "router_conf": router_prob[0, router_pred].item(),
                    # ── Uncertainty Extension ──
                    "uncertainty_mode": True,
                    "mc_samples": mc_samples,
                    "alpha": alpha,
                    "prediction_set": prediction_set,
                    "prediction_set_labels": [p["class"] for p in prediction_set],
                    "epistemic_uncertainty": epistemic_uncertainty,
                    "mean_probs": {class_names[i]: float(mean_probs[i]) for i in range(len(class_names))},
                    "variance": {class_names[i]: float(var_probs[i]) for i in range(len(class_names))},
                }
            
            else:
                # ─── Standard Deterministic Inference (Original) ─────
                logits = specialist(input_tensor)
                probs = torch.softmax(logits, dim=1)
                idx = torch.argmax(probs, dim=1).item()
                
                diagnosis = class_names[idx]
                confidence = probs[0, idx].item()
                
                top3_prob, top3_idx = torch.topk(probs, 3)
                top3 = {class_names[i]: p.item() for i, p in zip(top3_idx[0], top3_prob[0])}
                
                return {
                    "triage": path_name,
                    "diagnosis": diagnosis,
                    "confidence": confidence,
                    "top3": top3,
                    "router_conf": router_prob[0, router_pred].item()
                }
