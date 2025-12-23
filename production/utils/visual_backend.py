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
        self.model_path = os.path.join(base_path, models_dir, "Engine_A_ResNet-50.pth")
        self.csv_path = os.path.join(base_path, data_dir, "train_labels.csv")
        
        self.class_map = self._build_class_map()
        self.model = self._load_model()
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

    def _build_class_map(self):
        # Hardcoded to match training configuration
        # This removes dependency on external CSV file during inference
        unique_labels = sorted([
            'AV_Block', 'Atrial_Fibrillation', 'Fascicular_Block', 'IVCD', 'Ischemia', 
            'LBBB', 'Left_Hypertrophy', 'MI_Anterior', 'MI_Inferior', 'MI_Lateral', 
            'NORM', 'Paced', 'RBBB', 'Right_Hypertrophy', 'SVT', 'Sinus_Rhythm'
        ])
        return {i: label for i, label in enumerate(unique_labels)}

    def _load_model(self):
        num_classes = len(self.class_map)
        print(f"Visual Model: Loading ResNet50 with {num_classes} classes...")
        
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            # Handle strict=False in case of minor mismatches, but usually safe for full models
            model.load_state_dict(state_dict, strict=False) 
        else:
            print(f"Warning: Visual Model not found at {self.model_path}")
            
        model.to(self.device).eval()
        return model

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
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, idx = torch.max(probs, 1)
                
            label = self.class_map.get(idx.item(), "Unknown")
            
            # Top 3
            top3_prob, top3_idx = torch.topk(probs, 3)
            top3 = {self.class_map[i.item()]: p.item() for i, p in zip(top3_idx[0], top3_prob[0])}
            
            return {
                "diagnosis": label,
                "confidence": conf.item(),
                "top3": top3,
                "saved_files": save_paths
            }
            
        except Exception as e:
            return {"error": str(e)}
