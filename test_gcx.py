import os
import sys
import torch
import numpy as np

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'production'))

from utils.backend import CardiacPredictor, STRUCTURE_CLASSES
from utils.explainability.counterfactual import generate_counterfactual

def run_test():
    print("=" * 60)
    print("GCX Counterfactual Generation Validation test")
    print("=" * 60)
    
    predictor = CardiacPredictor(models_dir='models')
    device = predictor.device
    
    # We will target the structure net directly to isolate the GCX logic
    # Make sure we use the standard model, not wrapped in fusion if not active
    specialist = predictor.structure_net
    class_names = STRUCTURE_CLASSES
    
    # Generate a dummy signal
    np.random.seed(42)
    fake_ecg = np.random.randn(12, 5000).astype(np.float32) * 0.5
    input_tensor = predictor.preprocess(fake_ecg)
    
    # Find base prediction
    with torch.no_grad():
        logits_base = specialist(input_tensor)
        probs_base = torch.softmax(logits_base, dim=1)[0]
        pred_base = torch.argmax(probs_base).item()
        
    print(f"Baseline Prediction: {class_names[pred_base]} ({probs_base[pred_base]:.4f})")
    
    # Find a different target class (e.g. Normal or NDT)
    target_idx = -1
    for i, name in enumerate(class_names):
        if 'Normal' in name or 'NDT' in name:
            target_idx = i
            break
            
    if target_idx == -1:
        target_idx = (pred_base + 1) % len(class_names)  # Just pick another class if Normal not found
        
    print(f"Targeting Counterfactual: {class_names[target_idx]}")
    
    # Run optimizer
    print("Executing GCX (max_steps=100)...")
    cf_signal = generate_counterfactual(
        model=specialist,
        input_signal=input_tensor,
        target_class_idx=target_idx,
        epsilon=0.1,
        lambda_l2=0.05,
        lambda_tv=0.1,
        lr=0.05,  # Higher learning rate for strict convergence in tiny steps
        max_steps=100
    )
    
    # Re-evaluate
    with torch.no_grad():
        logits_cf = specialist(cf_signal)
        probs_cf = torch.softmax(logits_cf, dim=1)[0]
        pred_cf = torch.argmax(probs_cf).item()
        
    print(f"Counterfactual Result: {class_names[pred_cf]} ({probs_cf[pred_cf]:.4f})")
    
    # Difference stats
    delta = (cf_signal - input_tensor).abs()
    print(f"Max perturbation (L_inf bound check): {delta.max().item():.4f} (Epsilon=0.1)")
    print(f"Mean absolute perturbation (L1 check): {delta.mean().item():.4f}")
    
    if pred_cf == target_idx:
        print("\n✅ GCX successfully shifted prediction!")
    else:
        print("\n⚠️ GCX failed to shift prediction within 100 steps.")
        
    print("=" * 60)

if __name__ == "__main__":
    run_test()
