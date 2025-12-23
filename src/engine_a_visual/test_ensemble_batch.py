import torch
import pandas as pd
import argparse
import os
import sys
import random
from tqdm import tqdm

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ensemble import EnsembleEngine
from inference import group_diagnostic_classes

def test_batch(resnet_path, effnet_path, csv_path, image_dir, num_samples=50):
    print(f"--- Ensemble Batch Verification (N={num_samples}) ---")
    
    # 1. Load CSV
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    col = 'label' if 'label' in df.columns else 'diagnostic_superclass'
    
    # 2. Sample random rows
    # Filter for files that exist (optional, but good for safety)
    valid_indices = []
    print("Selecting random samples...")
    
    # Quick random sample to avoid scanning all 20k files if possible, 
    # but we need to ensure files exist.
    possible_indices = df.index.tolist()
    random.shuffle(possible_indices)
    
    samples = []
    attempts = 0
    max_attempts = num_samples * 5
    
    for idx in possible_indices:
        if len(samples) >= num_samples:
            break
        if attempts >= max_attempts:
            break
            
        row = df.iloc[idx]
        fname = str(row['filename'])
        if not fname.endswith('.png'): 
            fname += '.png'
            
        full_path = os.path.join(image_dir, fname)
        if os.path.exists(full_path):
            samples.append((full_path, str(row[col])))
        attempts += 1
            
    if not samples:
        print("Could not find valid images.")
        return

    # 3. Initialize Engine
    engine = EnsembleEngine(resnet_path, effnet_path, csv_path)
    
    # 4. Run Inference
    correct = 0
    total = len(samples)
    
    print(f"\nRunning Inference on {total} images...")
    
    print(f"{'FILENAME':<30} | {'TRUE (Grouped)':<20} | {'PRED':<20} | {'CONF':<8} | {'STATUS'}")
    print("-" * 100)
    
    for img_path, raw_label in samples:
        true_group = group_diagnostic_classes(raw_label)
        
        pred_label, conf = engine.predict(img_path)
        
        is_correct = (pred_label == true_group)
        if is_correct:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"
            
        basename = os.path.basename(img_path)
        print(f"{basename:<30} | {true_group:<20} | {pred_label:<20} | {conf:.4f}   | {status}")
        
    acc = (correct / total) * 100
    print("-" * 100)
    print(f"Batch Accuracy: {acc:.2f}% ({correct}/{total})")
    print("Note: This is a random subset. Full validation set accuracy is the gold standard.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet", default="models/classifier_resnet50.pth")
    parser.add_argument("--effnet", default="models/classifier_efficientnet_b4.pth")
    parser.add_argument("--csv", default="data/train_labels.csv")
    parser.add_argument("--images", default="data_synthesis/output/output/images/")
    parser.add_argument("--n", type=int, default=20)
    
    args = parser.parse_args()
    
    test_batch(args.resnet, args.effnet, args.csv, args.images, args.n)
