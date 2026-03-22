import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
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
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # production/
        self.router_path = os.path.join(base_path, models_dir, "prod_visual_router_v6.pth")
        self.structure_path = os.path.join(base_path, models_dir, "prod_visual_structure_v6.pth")
        self.rhythm_path = os.path.join(base_path, models_dir, "prod_visual_rhythm_v6.pth")
        
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

    def predict(self, image_input, patient_metadata=None):
        """
        image_input: PIL Image or path
        patient_metadata: dict with keys 'name', 'age', 'gender' (Optional)
        """
        # Pipeline: Smart Preprocess -> Tensor Norm
        transform = transforms.Compose([
            transforms.ToTensor(), # Standardizes to [0,1], CHW
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        try:
            # --- DATA MANAGEMENT ---
            # 0. Generate Filename if metadata provided
            save_paths = {}
            if patient_metadata:
                import time
                timestamp = int(time.time())
                sanitized_name = str(patient_metadata.get('name', 'Anonymous')).replace(" ", "_")
                age = str(patient_metadata.get('age', 'NA'))
                gender = str(patient_metadata.get('gender', 'NA'))
                
                # Format: Name_Age_Gender_Timestamp.png
                base_filename = f"{sanitized_name}_{age}_{gender}_{timestamp}"
                
                # Define Paths
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # production/
                raw_dir = os.path.join(base_path, "data", "engine_a", "patient_raw")
                pre_dir = os.path.join(base_path, "data", "engine_a", "raw_preprocessed")
                
                # Ensure directories exist (redundancy check)
                os.makedirs(raw_dir, exist_ok=True)
                os.makedirs(pre_dir, exist_ok=True)
                
                save_paths['raw'] = os.path.join(raw_dir, f"{base_filename}_raw.png")
                save_paths['pre'] = os.path.join(pre_dir, f"{base_filename}_processed.png")

                # Save RAW Upload
                if isinstance(image_input, str):
                    # It's a path, just copy or open-save
                    try:
                        Image.open(image_input).save(save_paths['raw'])
                    except: 
                        pass # Best effort
                elif isinstance(image_input, Image.Image):
                    image_input.save(save_paths['raw'])
            
            # 1. Smart Preprocessing (Crop -> Enhance -> Resize/Pad)
            # We match the IMAGE_SIZE constant (512) for the model
            processed_numpy = self.preprocessor.process(image_input, target_size=IMAGE_SIZE)
            
            # Save PROCESSED Image
            if 'pre' in save_paths:
                # processed_numpy is RGB, convert to BGR for cv2 or just use PIL
                Image.fromarray(processed_numpy).save(save_paths['pre'])

            # 2. Convert to PIL for Transforms (or use directly if tensor supports it)
            # processed_numpy is RGB uint8
            pil_image = Image.fromarray(processed_numpy)
             
            tensor = transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # L2: Router
                router_out = self.router(tensor)
                router_probs = torch.softmax(router_out, dim=1)
                router_conf, router_idx = torch.max(router_probs, 1)
                domain = self.router_classes[router_idx.item()]
                
                # L3: Specialist
                if domain == 'Structure':
                    spec_out = self.structure_net(tensor)
                    class_map = self.structure_classes
                else:
                    spec_out = self.rhythm_net(tensor)
                    class_map = self.rhythm_classes
                    
                spec_probs = torch.softmax(spec_out, dim=1)
                spec_conf, spec_idx = torch.max(spec_probs, 1)
                
            label = class_map.get(spec_idx.item(), "Unknown")
            
            # Top 3 from the chosen specialist
            top3_prob, top3_idx = torch.topk(spec_probs, min(3, len(class_map)))
            top3 = {class_map[i.item()]: p.item() for i, p in zip(top3_idx[0], top3_prob[0])}
            
            # Combine confidence (Router Confidence * Specialist Confidence)
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
