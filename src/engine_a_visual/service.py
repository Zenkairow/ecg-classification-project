import os
import torch
import traceback
from PIL import Image

# Internal modules
from .ensemble import EnsembleEngine
from .preprocess import ECGScanner
from .inference import group_diagnostic_classes

class VisualDiagnosisService:
    """
    The High-Level Service that the Website Backend (Flask/FastAPI) will call.
    Orchestrates:
    1. Preprocessing (Scanning, De-warping, Shadow Removal)
    2. Ensemble Inference (ResNet-50 + EfficientNet-B4)
    3. Result Formatting
    """
    def __init__(self, models_dir="models", data_dir="data"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Initializing Visual Engine Service [{self.device}] ---")
        
        # Paths
        self.resnet_path = os.path.join(models_dir, "classifier_resnet50.pth")
        self.effnet_path = os.path.join(models_dir, "classifier_efficientnet_b4.pth")
        self.csv_path = os.path.join(data_dir, "train_labels.csv")
        
        # Components
        print("1. Loading Ensemble Models...")
        self.engine = EnsembleEngine(self.resnet_path, self.effnet_path, self.csv_path)
        
        print("2. Initializing Preprocessor...")
        # Target size 1024 to feed the High-Res EfficientNet properly
        self.scanner = ECGScanner(target_size=(1024, 1024)) 
        
        print("--- Visual Engine Ready ---")

    def process_request(self, image_path_or_file):
        """
        Main entry point for the website.
        Args:
            image_path_or_file: Path to image or file-like object.
        Returns:
            dict: {
                'diagnosis': str, 
                'confidence': float, 
                'status': 'success'|'error',
                'meta': {...}
            }
        """
        response = {
            "diagnosis": "Unknown",
            "confidence": 0.0,
            "status": "error",
            "meta": {}
        }
        
        try:
            # Step 1: Preprocessing
            # We assume image_path_or_file is a filepath for now. 
            # If it's a stream, ECGScanner might need adaptation, but usually PIL handles both.
            processed_image = self.scanner.preprocess(image_path_or_file)
            
            # Step 2: Prediction
            # The engine expects a path, but we have a PIL Image now.
            # We need to adapt EnsembleEngine to accept PIL images if it doesn't already using `predict_image` logic.
            # Let's assume we update EnsembleEngine to handle PIL, or we save a temp file.
            # For Performance, passing PIL is better.
            
            # Check if engine supports PIL (we will ensure it does)
            raw_label, conf = self.engine.predict_pil(processed_image)
            
            response["diagnosis"] = raw_label # Already grouped by EnsembleEngine
            response["confidence"] = float(conf)
            response["status"] = "success"
            response["meta"] = {
                "model": "Ensemble (ResNet50+EfficientNetB4)",
                "preprocessing": "CLAHE + PerspectiveWarp"
            }
            
        except Exception as e:
            print(f"Service Error: {e}")
            traceback.print_exc()
            response["error_message"] = str(e)
            
        return response

# Example Usage for Testing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to test image")
    args = parser.parse_args()
    
    # Initialize Service
    service = VisualDiagnosisService()
    
    # Run
    result = service.process_request(args.image)
    print("\nAPI Response:")
    print(result)
