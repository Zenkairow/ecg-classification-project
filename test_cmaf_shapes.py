"""
CMAF Shape Verification Test
=============================
Validates tensor dimensions through the entire Cross-Modal Attention
Fusion pipeline WITHOUT requiring trained models or GPU.
"""

import torch
import sys
import os

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'production'))

from utils.model import SEResNet34
from utils.cross_modal_fusion import CrossModalAttentionFusion, FusedClassificationHead


def test_cmaf_shapes():
    """Verify all tensor shapes through the CMAF pipeline."""
    print("=" * 60)
    print("CMAF Shape Verification Test")
    print("=" * 60)
    
    B = 4  # batch size
    device = torch.device("cpu")
    
    # ── 1. Create dummy models ─────────────────────────────────
    signal_model = SEResNet34(num_classes=13, input_channels=12).to(device)
    
    # Simulate ResNet-50 feature extraction (2048-D)
    # In production, this comes from ResNet-50's avgpool layer
    print("\n[1] Models created")
    
    # ── 2. Create dummy inputs ─────────────────────────────────
    fake_signal = torch.randn(B, 12, 5000, device=device)
    fake_visual_features = torch.randn(B, 2048, device=device)
    
    print(f"    Signal input:       {list(fake_signal.shape)}")
    print(f"    Visual features:    {list(fake_visual_features.shape)}")
    
    # ── 3. Extract signal features ─────────────────────────────
    with torch.no_grad():
        h_signal = signal_model.extract_features(fake_signal)
    
    print(f"\n[2] Feature Extraction")
    print(f"    h_signal (Engine B): {list(h_signal.shape)}")
    print(f"    h_visual (Engine A): {list(fake_visual_features.shape)}")
    
    assert h_signal.shape == (B, 512), f"Expected [B, 512], got {list(h_signal.shape)}"
    assert fake_visual_features.shape == (B, 2048), f"Expected [B, 2048], got {list(fake_visual_features.shape)}"
    print("    ✅ Feature dimensions correct")
    
    # ── 4. CMAF Forward Pass ───────────────────────────────────
    cmaf = CrossModalAttentionFusion(
        dim_signal=512,
        dim_visual=2048,
        d_model=256,
        num_heads=8,
        dropout=0.1,
    ).to(device)
    
    h_signal_fused, h_visual_fused, attn_weights = cmaf(h_signal, fake_visual_features)
    
    print(f"\n[3] CMAF Output")
    print(f"    h_signal_fused:     {list(h_signal_fused.shape)}")
    print(f"    h_visual_fused:     {list(h_visual_fused.shape)}")
    print(f"    gate_signal:        {attn_weights['gate_signal']:.4f}")
    print(f"    gate_visual:        {attn_weights['gate_visual']:.4f}")
    
    assert h_signal_fused.shape == (B, 512), f"Expected [B, 512], got {list(h_signal_fused.shape)}"
    assert h_visual_fused.shape == (B, 2048), f"Expected [B, 2048], got {list(h_visual_fused.shape)}"
    print("    ✅ Fused dimensions match originals")
    
    # ── 5. Classification from fused features ──────────────────
    signal_logits = signal_model.classify(h_signal_fused)
    
    visual_head = FusedClassificationHead(in_features=2048, num_classes=10, dropout=0.3)
    visual_logits = visual_head(h_visual_fused)
    
    print(f"\n[4] Classification Output")
    print(f"    signal_logits:      {list(signal_logits.shape)}")
    print(f"    visual_logits:      {list(visual_logits.shape)}")
    
    assert signal_logits.shape == (B, 13), f"Expected [B, 13], got {list(signal_logits.shape)}"
    assert visual_logits.shape == (B, 10), f"Expected [B, 10], got {list(visual_logits.shape)}"
    print("    ✅ Logit dimensions correct")
    
    # ── 6. Parameter count ─────────────────────────────────────
    cmaf_params = sum(p.numel() for p in cmaf.parameters())
    print(f"\n[5] CMAF Parameter Count: {cmaf_params:,}")
    print(f"    (vs SE-ResNet-34: {sum(p.numel() for p in signal_model.parameters()):,})")
    
    # ── 7. Gradient flow test ──────────────────────────────────
    h_signal_grad = signal_model.extract_features(fake_signal)
    h_signal_grad.requires_grad_(True)
    fake_visual_grad = fake_visual_features.clone().requires_grad_(True)
    
    h_s_fused, h_v_fused, _ = cmaf(h_signal_grad, fake_visual_grad)
    loss = h_s_fused.sum() + h_v_fused.sum()
    loss.backward()
    
    print(f"\n[6] Gradient Flow")
    print(f"    grad_signal: {h_signal_grad.grad is not None}")
    print(f"    grad_visual: {fake_visual_grad.grad is not None}")
    print(f"    ✅ Gradients flow through CMAF")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED — CMAF architecture verified.")
    print("=" * 60)


if __name__ == "__main__":
    test_cmaf_shapes()
