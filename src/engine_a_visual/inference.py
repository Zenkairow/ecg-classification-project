import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import argparse
import os
import sys

# --- Constants ---
IMAGE_SIZE = 512

def load_model_and_classes(model_path, csv_path):
    """
    Loads the ResNet50/EfficientNet model and reconstructs the class mapping from the CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Rebuild Class Map
    if not os.path.exists(csv_path):
        print(f"Error: Labels CSV not found at {csv_path}. Cannot decode classes.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    # Handle column names
    col_name = 'label' if 'label' in df.columns else 'diagnostic_superclass'
    if col_name not in df.columns:
         print("Error: CSV must have a 'label' or 'diagnostic_superclass' column.")
         sys.exit(1)
        
    unique_labels = sorted(df[col_name].unique().tolist())
    class_map = {i: label for i, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    
    print(f"Detected {num_classes} classes from CSV.")

    # 2. Initialize Model Architecture
    # Try loading as EfficientNet first, if it fails (state dict mismatch), fallback or default
    try:
        print("Attempting to load as EfficientNet-B4...")
        model = models.efficientnet_b4(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, num_classes),
        )
    except:
        print("Fallback to ResNet50...")
        model = models.resnet50(weights=None) 
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 3. Load State Dict
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
        
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights (Architecture mismatch?): {e}")
        # If EfficientNet failed, maybe it's ResNet
        try:
            print("Retrying with ResNet50 architecture...")
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            model.load_state_dict(state_dict)
            print("ResNet model loaded successfully.")
        except Exception as e2:
             print(f"Fatal Error: {e2}")
             sys.exit(1)
        
    model = model.to(device)
    model.eval()
    
    return model, class_map, device

def predict_image(image_input, model, class_map, device):
    """
    Runs inference. image_input can be a filepath (str) or a PIL Image object.
    """
    # Preprocessing (Must match training!)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                 print(f"Error: Image not found at {image_input}")
                 return None, 0.0
            image = Image.open(image_input).convert('RGB')
        else:
            # Assume it's a PIL Object
            image = image_input.convert('RGB')
            
        image_tensor = transform(image).unsqueeze(0) # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        idx = predicted_idx.item()
        label = class_map.get(idx, "Unknown")
        conf_score = confidence.item()
        
        return label, conf_score
        
    except Exception as e:
        print(f"Inference Error: {e}")
        return None, 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Visual Classifier Inference")
    parser.add_argument("image_input", type=str, help="Path to the ECG image")
    parser.add_argument("--model", type=str, default="models/classifier_efficientnet_b4.pth", help="Path to trained model checkpoint")
    parser.add_argument("--csv", type=str, default="data/train_labels.csv", help="Path to training CSV (for class mapping)")
    parser.add_argument("--preprocess", action="store_true", help="Enable Real-World Preprocessing (Scan + CLAHE)")
    
    args = parser.parse_args()
    
    print("--- ECG Inference Engine ---")
    model, class_map, device = load_model_and_classes(args.model, args.csv)
    
    if args.preprocess:
        print("PREPROCESSING: Enabling Safe Scanner (Warp + CLAHE)...")
        try:
            from preprocess import ECGScanner
            scanner = ECGScanner(target_size=(IMAGE_SIZE, IMAGE_SIZE))
            # Preprocess to PIL Image
            image_pil = scanner.preprocess(args.image_input)
            image_pil.save("debug_preprocessed.png")
            print("Debug: Saved 'debug_preprocessed.png'")
            final_input = image_pil
        except ImportError:
             print("Error: src/preprocess.py not found.")
             sys.exit(1)
        except Exception as e:
             print(f"Preprocessing Failed: {e}")
             sys.exit(1)
    else:
        final_input = args.image_input

    print(f"\nAnalyzing...")
    label, conf = predict_image(final_input, model, class_map, device)
    
    if label:
        print(f"\n>>> DIAGNOSIS: {label}")
        print(f">>> CONFIDENCE: {conf:.4f} ({conf*100:.2f}%)")
    else:
        print("\n>>> DIAGNOSIS FAILED")
