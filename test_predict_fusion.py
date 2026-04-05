"""
CMAF End-to-End Pipeline Validation
===================================
Tests the predict_fusion pipeline using random tensors directly
against the modified CardiacSystem.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
production_dir = os.path.join(current_dir, 'production')
sys.path.append(production_dir)

from utils.consensus import CardiacSystem

def run_fusion_test():
    print("=" * 60)
    print("Initializing CardiacSystem for CMAF Validation...")
    print("=" * 60)
    
    # 1. Initialize System
    # Note: If real visual models are missing, it will print warnings but still create the ResNet structures internally.
    # However, visual_available relies on strict loading. For test purposes, we'll force it available if models load with warnings.
    system = CardiacSystem(models_dir='models')
    
    # Force engines available for shape-testing if they failed due to missing files
    if not system.signal_available:
        print("[!] Forcing Signal Engine online for test.")
        system.signal_available = True
    if not system.visual_available:
        print("[!] Forcing Visual Engine online for test.")
        system.visual_available = True
    if not system.fusion_available:
        print("[!] Forcing Fusion Engine online for test.")
        system.fusion_available = True
        
    print("\n" + "=" * 60)
    print("Executing predict_fusion() with Synthetic Data...")
    
    # 2. Create Dummy Data
    np.random.seed(42)
    dummy_signal = np.random.randn(12, 5000).astype(np.float32)
    dummy_image = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))
    
    # 3. Predict Fusion
    result = system.predict_fusion(dummy_signal, dummy_image)
    
    # 4. Results
    print("\n--- CMAF Integration Result ---")
    if "error" in result:
        print(f"ERROR: {result['error']}")
        if "message" in result:
            print(f"Message: {result['message']}")
    else:
        print(f"Fused Diagnosis: {result['diagnosis']} (Confidence: {result['confidence']:.4f})")
        print(f"Target Domain:   {result['domain']}")
        print(f"Signal Gate:     {result['attention_gate_signal']:.4f}")
        print(f"Visual Gate:     {result['attention_gate_visual']:.4f}")
        print("\nFused Probabilities:")
        for k, v in result['fused_probs'].items():
            print(f"  {k:20s}: {v:.4f}")
            
    print("\n" + "=" * 60)
    print("TEST COMPLETED.")
    print("=" * 60)

if __name__ == "__main__":
    run_fusion_test()
