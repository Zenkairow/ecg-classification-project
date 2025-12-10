import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import argparse
import os
import sys
import torch.nn.functional as F
from inference import group_diagnostic_classes  # Reuse logic

# --- Constants ---
IMAGE_SIZE_RESNET = 512
IMAGE_SIZE_EFFNET = 1024

class EnsembleEngine:
    def __init__(self, resnet_path, effnet_path, csv_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.csv_path = csv_path
        
        # 1. Setup Maps
        self.map_50, self.map_20 = self._build_class_maps()
        self.num_classes_50 = len(self.map_50)
        self.num_classes_20 = len(self.map_20)
        
        # 2. Load Models
        self.resnet = self._load_resnet(resnet_path)
        self.effnet = self._load_effnet(effnet_path)
        
    def _build_class_maps(self):
        """
        Builds both the 50-class raw map (for ResNet) and 20-class grouped map (for EffNet).
        """
        df = pd.read_csv(self.csv_path)
        col = 'label' if 'label' in df.columns else 'diagnostic_superclass'
        
        # Map 50 (Raw)
        raw_labels = sorted(df[col].unique().tolist())
        map_50 = {i: label for i, label in enumerate(raw_labels)}
        
        # Map 20 (Grouped)
        grouped_labels = set()
        for l in raw_labels:
            grouped_labels.add(group_diagnostic_classes(str(l)))
        unique_grouped = sorted(list(grouped_labels))
        map_20 = {i: label for i, label in enumerate(unique_grouped)}
        
        # Invert Map 20 for lookup (Label -> Index)
        self.map_20_inv = {label: i for i, label in map_20.items()}
        
        return map_50, map_20

    def _load_resnet(self, path):
        print(f"Loading ResNet-50 (50 Classes) from {path}...")
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes_50)
        
        try:
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading ResNet: {e}")
            sys.exit(1)
            
        model = model.to(self.device)
        model.eval()
        return model

    def _load_effnet(self, path):
        print(f"Loading EfficientNet-B4 (20 Classes) from {path}...")
        model = models.efficientnet_b4(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, self.num_classes_20),
        )
        
        try:
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading EfficientNet: {e}")
            sys.exit(1)
            
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path):
        if not os.path.exists(image_path):
            print("Image not found.")
            return None
            
        img = Image.open(image_path).convert('RGB')
        
        # --- ResNet Prediction (512px) ---
        t_resnet = transforms.Compose([
            transforms.Resize((IMAGE_SIZE_RESNET, IMAGE_SIZE_RESNET)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_img_res = t_resnet(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out_res = self.resnet(t_img_res)
            probs_res_50 = F.softmax(out_res, dim=1).cpu().numpy()[0]
            
        # --- EfficientNet Prediction (1024px) ---
        t_effnet = transforms.Compose([
            transforms.Resize((IMAGE_SIZE_EFFNET, IMAGE_SIZE_EFFNET)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_img_eff = t_effnet(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out_eff = self.effnet(t_img_eff)
            probs_eff_20 = F.softmax(out_eff, dim=1).cpu().numpy()[0]

        # --- Aggregation (The Magic) ---
        # Initialize 20-class vector for ResNet
        probs_res_20_aggregated = torch.zeros(self.num_classes_20)
        
        for i, prob in enumerate(probs_res_50):
            raw_label = self.map_50[i]
            target_group = group_diagnostic_classes(str(raw_label))
            target_idx = self.map_20_inv[target_group]
            probs_res_20_aggregated[target_idx] += prob
            
        # Weighted Average (Give EffNet slightly more weight as it is High-Res)
        # Weighting: 0.6 EffNet + 0.4 ResNet
        final_probs = (0.6 * torch.tensor(probs_eff_20)) + (0.4 * probs_res_20_aggregated)
        
        best_conf, best_idx = torch.max(final_probs, 0)
        final_label = self.map_20[best_idx.item()]
        
        return final_label, best_conf.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Inference")
    parser.add_argument("image", type=str)
    # Defaults based on server paths
    parser.add_argument("--resnet", default="models/classifier_resnet50.pth")
    parser.add_argument("--effnet", default="models/classifier_efficientnet_b4.pth")
    parser.add_argument("--csv", default="data/train_labels.csv")
    
    args = parser.parse_args()
    
    engine = EnsembleEngine(args.resnet, args.effnet, args.csv)
    label, conf = engine.predict(args.image)
    
    print(f"\n>>> ENSEMBLE DIAGNOSIS: {label}")
    print(f">>> CONFIDENCE: {conf:.4f}")
