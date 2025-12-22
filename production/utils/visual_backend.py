import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
import sys

# Constants
IMAGE_SIZE = 512

class VisualPredictor:
    def __init__(self, models_dir="models", data_dir="data"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # production/
        self.model_path = os.path.join(base_path, models_dir, "Engine_A_ResNet-50.pth")
        self.csv_path = os.path.join(base_path, data_dir, "train_labels.csv")
        
        self.class_map = self._build_class_map()
        self.model = self._load_model()
        
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
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found at {self.csv_path}")
            
        df = pd.read_csv(self.csv_path)
        col = 'label' if 'label' in df.columns else 'diagnostic_superclass'
        
        raw_labels = df[col].unique().tolist()
        grouped_labels = set()
        for l in raw_labels:
            grouped_labels.add(self._group_diagnostic_classes(l))
            
        unique_labels = sorted(list(grouped_labels))
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

    def predict(self, image_input):
        """
        image_input: PIL Image or path
        """
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
            else:
                image = image_input.convert('RGB')
                
            tensor = transform(image).unsqueeze(0).to(self.device)
            
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
                "top3": top3
            }
            
        except Exception as e:
            return {"error": str(e)}
