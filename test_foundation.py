import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from models_foundation import ECGFoundationWrapper

def test_foundation_wrapper():
    print("=" * 60)
    print("Testing Foundation Model Setup (v3.5-experimental)")
    print("=" * 60)
    
    # 1. Instantiate the Wrapper
    print("\n[1] Instantiating ECGFoundationWrapper...")
    model = ECGFoundationWrapper(
        input_channels=12,
        seq_len=5000,
        d_model=256,
        nhead=8,
        num_layers=6
    )
    
    # 2. Freeze the backbone
    print("[2] Executing self.freeze_backbone()...")
    model.freeze_backbone()
    
    # Verify freeze was successful
    frozen = all(not p.requires_grad for p in model.backbone.parameters())
    print(f"    Backbone correctly frozen: {frozen}")
    
    # 3. Validation Pipeline execution with dummy tensor
    # Simulate a batched [B, 12, L] continuous timeseries sample
    B = 2
    dummy_input = torch.randn(B, 12, 5000)
    print(f"\n[3] Ingesting Mock Tensor => Shape: {list(dummy_input.shape)}")
    
    # Run through the distinct classification heads
    with torch.no_grad():
        router_logits = model(dummy_input, task='router')
        structure_logits = model(dummy_input, task='structure')
        rhythm_logits = model(dummy_input, task='rhythm')
        
    # Check outputs
    print("\n--- Model Head Output Vector Shapes ---")
    print(f"Router    [target: [2, 2]]   => {list(router_logits.shape)}")
    print(f"Structure [target: [2, 11]]  => {list(structure_logits.shape)}")
    print(f"Rhythm    [target: [2, 13]]  => {list(rhythm_logits.shape)}")
    
    assert list(router_logits.shape) == [B, 2]
    assert list(structure_logits.shape) == [B, 11]
    assert list(rhythm_logits.shape) == [B, 13]
    
    print("\n✅ All shapes validated successfully.")
    print("=" * 60)

if __name__ == "__main__":
    test_foundation_wrapper()
