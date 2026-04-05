import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import sys

# Constants
IMAGE_SIZE = 512

from utils.preprocessing import SmartImagePreprocessor

class VisualPredictor:
    def __init__(self, models_dir="models", data_dir="data"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # project root
        self.router_path = os.path.join(project_root, models_dir, "prod_visual_router_v6.pth")
        self.structure_path = os.path.join(project_root, models_dir, "prod_visual_structure_v6.pth")
        self.rhythm_path = os.path.join(project_root, models_dir, "prod_visual_rhythm_v6.pth")
        
        self.router_classes = {0: 'Rhythm', 1: 'Structure'}
        
        # Hardcoded Specialist Classes to match V3 training
        self.structure_classes = {
            0: 'IVCD', 1: 'Ischemia', 2: 'LBBB', 3: 'LNGQT', 4: 'Left_Hypertrophy',
            5: 'MI_Anterior', 6: 'MI_Inferior', 7: 'MI_Lateral', 8: 'RBBB', 9: 'Right_Hypertrophy'
        }
        
        self.rhythm_classes = {
            0: 'AV_Block', 1: 'Atrial_Fibrillation', 2: 'DIG', 3: 'EL', 4: 'Fascicular_Block',
            5: 'NDT', 6: 'PVC', 7: 'Paced', 8: 'SVT', 9: 'Sinus_Rhythm', 10: 'WPW'
        }
        
        self._load_models()
        self.preprocessor = SmartImagePreprocessor()
        
    def _group_diagnostic_classes(self, label):
        """Standardizes raw labels into clinical categories."""
        label = str(label)
        if label in ['AMI', 'ALMI', 'ASMI', 'INJAL', 'INJAS']: return 'MI_Anterior'
        if label in ['IMI', 'ILMI', 'IPLMI', 'IPMI', 'INJIL', 'INJIN']: return 'MI_Inferior'
        if label in ['LMI', 'INJLA', 'PMI']: return 'MI_Lateral'
        if 'ISC' in label or label == 'NST_': return 'Ischemia'
        if label in ['CLBBB', 'ILBBB']: return 'LBBB'
        if label in ['CRBBB', 'IRBBB']: return 'RBBB'
        if label == 'IVCD': return 'IVCD'
        if label in ['1AVB', '2AVB', '3AVB']: return 'AV_Block'
        if label in ['LVH', 'LAO/LAE']: return 'Left_Hypertrophy'
        if label in ['RVH', 'RAO/RAE', 'SEHYP']: return 'Right_Hypertrophy'
        if label in ['LAFB', 'LPFB']: return 'Fascicular_Block'
        if label in ['AFIB', 'AFLT']: return 'Atrial_Fibrillation'
        if label in ['SARRH', 'STACH', 'SBRAD', 'SR']: return 'Sinus_Rhythm'
        if label == 'PACE': return 'Paced'
        if label in ['PSVT', 'SVT']: return 'SVT'
        if label == 'NORM': return 'NORM'
        return label

    def _load_single_model(self, path, num_classes):
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Visual Model not found at {path}")
        model.to(self.device).eval()
        return model

    def _load_models(self):
        print("Visual Hierarchy: Loading Router and Specialists...")
        self.router = self._load_single_model(self.router_path, 2)
        self.structure_net = self._load_single_model(self.structure_path, 10)
        self.rhythm_net = self._load_single_model(self.rhythm_path, 11)

    # -- MC-Dropout Utilities ------------------------------------------------
    @staticmethod
    def _enable_mc_dropout(model):
        """Force Dropout layers to remain active during inference."""
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @staticmethod
    def _disable_mc_dropout(model):
        """Restore all Dropout layers to eval mode."""
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

    def predict(self, image_input, patient_metadata=None, uncertainty_mode=False, mc_samples=50, alpha=0.1):
        """
        image_input: PIL Image or path
        patient_metadata: dict with keys 'name', 'age', 'gender' (Optional)
        uncertainty_mode: If True, enable MC-Dropout Conformal Prediction.
        mc_samples: Number of stochastic forward passes (T). Default: 50.
        alpha: Significance level for prediction set. Default: 0.1 (90% coverage).
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        try:
            # --- DATA MANAGEMENT ---
            save_paths = {}
            if patient_metadata:
                import time
                timestamp = int(time.time())
                sanitized_name = str(patient_metadata.get('name', 'Anonymous')).replace(" ", "_")
                age = str(patient_metadata.get('age', 'NA'))
                gender = str(patient_metadata.get('gender', 'NA'))
                base_filename = f"{sanitized_name}_{age}_{gender}_{timestamp}"
                
                prod_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                raw_dir = os.path.join(prod_path, "data", "engine_a", "patient_raw")
                pre_dir = os.path.join(prod_path, "data", "engine_a", "raw_preprocessed")
                os.makedirs(raw_dir, exist_ok=True)
                os.makedirs(pre_dir, exist_ok=True)
                
                save_paths['raw'] = os.path.join(raw_dir, f"{base_filename}_raw.png")
                save_paths['pre'] = os.path.join(pre_dir, f"{base_filename}_processed.png")

                if isinstance(image_input, str):
                    try:
                        Image.open(image_input).save(save_paths['raw'])
                    except: 
                        pass
                elif isinstance(image_input, Image.Image):
                    image_input.save(save_paths['raw'])
            
            # 1. Smart Preprocessing
            processed_numpy = self.preprocessor.process(image_input, target_size=IMAGE_SIZE)
            
            if 'pre' in save_paths:
                Image.fromarray(processed_numpy).save(save_paths['pre'])

            pil_image = Image.fromarray(processed_numpy)
            tensor = transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # L2: Router (always deterministic)
                router_out = self.router(tensor)
                router_probs = torch.softmax(router_out, dim=1)
                router_conf, router_idx = torch.max(router_probs, 1)
                domain = self.router_classes[router_idx.item()]
                
                # Select specialist
                if domain == 'Structure':
                    specialist = self.structure_net
                    class_map = self.structure_classes
                else:
                    specialist = self.rhythm_net
                    class_map = self.rhythm_classes
                
                if uncertainty_mode:
                    # --- MC-Dropout Conformal Prediction ---
                    self._enable_mc_dropout(specialist)
                    
                    mc_probs_list = []
                    for _ in range(mc_samples):
                        spec_out = specialist(tensor)
                        spec_probs = torch.softmax(spec_out, dim=1)
                        mc_probs_list.append(spec_probs.cpu().numpy()[0])
                    
                    self._disable_mc_dropout(specialist)
                    
                    mc_probs = np.array(mc_probs_list)  # [T, C]
                    mean_probs = mc_probs.mean(axis=0)
                    var_probs = mc_probs.var(axis=0)
                    epistemic_uncertainty = float(var_probs.mean())
                    
                    top1_idx = int(np.argmax(mean_probs))
                    label = class_map.get(top1_idx, "Unknown")
                    spec_conf = float(mean_probs[top1_idx])
                    overall_confidence = router_conf.item() * spec_conf
                    
                    # Conformal Prediction Set
                    prediction_set = []
                    for i, p in enumerate(mean_probs):
                        if p >= alpha:
                            prediction_set.append({
                                "class": class_map.get(i, f"Class_{i}"),
                                "mean_prob": float(p),
                                "variance": float(var_probs[i]),
                                "std": float(np.sqrt(var_probs[i])),
                            })
                    prediction_set.sort(key=lambda x: x["mean_prob"], reverse=True)
                    
                    top3_indices = np.argsort(mean_probs)[::-1][:min(3, len(class_map))]
                    top3 = {class_map.get(i, f"Class_{i}"): float(mean_probs[i]) for i in top3_indices}
                    
                    return {
                        "diagnosis": label,
                        "confidence": overall_confidence,
                        "domain": domain,
                        "domain_confidence": router_conf.item(),
                        "top3": top3,
                        "saved_files": save_paths,
                        # -- Uncertainty Extension --
                        "uncertainty_mode": True,
                        "mc_samples": mc_samples,
                        "alpha": alpha,
                        "prediction_set": prediction_set,
                        "prediction_set_labels": [p["class"] for p in prediction_set],
                        "epistemic_uncertainty": epistemic_uncertainty,
                        "mean_probs": {class_map.get(i, f"Class_{i}"): float(mean_probs[i]) for i in range(len(class_map))},
                        "variance": {class_map.get(i, f"Class_{i}"): float(var_probs[i]) for i in range(len(class_map))},
                    }
                    
                else:
                    # --- Standard Deterministic Inference (Original) ---
                    spec_out = specialist(tensor)
                    spec_probs = torch.softmax(spec_out, dim=1)
                    spec_conf, spec_idx = torch.max(spec_probs, 1)
                    
                    label = class_map.get(spec_idx.item(), "Unknown")
                    top3_prob, top3_idx = torch.topk(spec_probs, min(3, len(class_map)))
                    top3 = {class_map[i.item()]: p.item() for i, p in zip(top3_idx[0], top3_prob[0])}
                    overall_confidence = router_conf.item() * spec_conf.item()
                    
                    return {
                        "diagnosis": label,
                        "confidence": overall_confidence,
                        "domain": domain,
                        "domain_confidence": router_conf.item(),
                        "top3": top3,
                        "saved_files": save_paths
                    }
            
        except Exception as e:
            return {"error": str(e)}
