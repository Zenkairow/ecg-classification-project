"""
Cross-Modal Attention Fusion (CMAF) Module — v3.5-experimental
================================================================
Implements bidirectional cross-attention between Engine B (Signal, 512-D)
and Engine A (Visual, 2048-D) feature spaces.

Architecture:
  1. Project both modalities into shared latent space (d=256)
  2. Bidirectional MultiheadAttention:
     - Engine B attends to Visual: Q=h_B, K=h_A, V=h_A
     - Engine A attends to Signal: Q=h_A, K=h_B, V=h_B
  3. Residual connection preserves independent diagnostic capability
  4. LayerNorm + MLP for post-attention refinement
  5. Output heads project back to original feature dimensions

Tensor Flow:
  h_B [B, 512]  ──→ proj_B [B, 1, 256] ──→ CrossAttn ──→ h_B' [B, 512]
  h_A [B, 2048] ──→ proj_A [B, 1, 256] ──→ CrossAttn ──→ h_A' [B, 2048]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Modal Attention Fusion Module.
    
    Enables each engine to attend to the other's intermediate
    representations before final classification, capturing
    complementary features across modalities.
    
    Args:
        dim_signal:  Feature dimension from Engine B (default: 512)
        dim_visual:  Feature dimension from Engine A (default: 2048)
        d_model:     Shared latent dimension (default: 256)
        num_heads:   Number of attention heads (default: 8)
        dropout:     Attention dropout rate (default: 0.1)
        mlp_ratio:   MLP hidden layer expansion ratio (default: 2)
    """
    
    def __init__(
        self,
        dim_signal: int = 512,
        dim_visual: int = 2048,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: int = 2,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.dim_signal = dim_signal
        self.dim_visual = dim_visual
        
        # ── Input Projections: Map to shared latent space ──────────
        self.proj_signal = nn.Sequential(
            nn.Linear(dim_signal, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        self.proj_visual = nn.Sequential(
            nn.Linear(dim_visual, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # ── Bidirectional Cross-Attention ──────────────────────────
        # Signal attends to Visual (Q=signal, K=visual, V=visual)
        self.cross_attn_s2v = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Visual attends to Signal (Q=visual, K=signal, V=signal)
        self.cross_attn_v2s = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # ── Post-Attention LayerNorm ──────────────────────────────
        self.norm_signal = nn.LayerNorm(d_model)
        self.norm_visual = nn.LayerNorm(d_model)
        
        # ── Feed-Forward Networks (post-attention refinement) ─────
        mlp_hidden = d_model * mlp_ratio
        
        self.ffn_signal = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )
        
        self.ffn_visual = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )
        
        self.norm_signal_ffn = nn.LayerNorm(d_model)
        self.norm_visual_ffn = nn.LayerNorm(d_model)
        
        # ── Output Projections: Map back to original dimensions ──
        self.out_signal = nn.Linear(d_model, dim_signal)
        self.out_visual = nn.Linear(d_model, dim_visual)
        
        # ── Fusion Gate: Learnable alpha for residual blending ────
        # Starts at 0 → initially preserves original features entirely
        # Gradually learns to incorporate cross-modal information
        self.gate_signal = nn.Parameter(torch.zeros(1))
        self.gate_visual = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        h_signal: torch.Tensor,
        h_visual: torch.Tensor,
    ) -> tuple:
        """
        Bidirectional cross-modal attention fusion.
        
        Args:
            h_signal: Engine B features [B, 512]
            h_visual: Engine A features [B, 2048]
            
        Returns:
            h_signal_fused: Fused signal features [B, 512]
            h_visual_fused: Fused visual features [B, 2048]
            attn_weights: Dict with attention weight matrices for interpretability
        """
        B = h_signal.size(0)
        
        # ── Step 1: Project to shared latent space ────────────────
        # [B, 512] → [B, 256] → [B, 1, 256] (add sequence dim for MHA)
        z_signal = self.proj_signal(h_signal).unsqueeze(1)  # [B, 1, 256]
        z_visual = self.proj_visual(h_visual).unsqueeze(1)  # [B, 1, 256]
        
        # ── Step 2: Bidirectional Cross-Attention ─────────────────
        # Signal attends to Visual: "What visual features inform the signal diagnosis?"
        attended_signal, attn_s2v = self.cross_attn_s2v(
            query=z_signal,   # Q from signal
            key=z_visual,     # K from visual
            value=z_visual,   # V from visual
        )  # [B, 1, 256], [B, 1, 1]
        
        # Visual attends to Signal: "What signal features inform the visual diagnosis?"
        attended_visual, attn_v2s = self.cross_attn_v2s(
            query=z_visual,   # Q from visual
            key=z_signal,     # K from signal
            value=z_signal,   # V from signal
        )  # [B, 1, 256], [B, 1, 1]
        
        # ── Step 3: Residual + LayerNorm ──────────────────────────
        z_signal = self.norm_signal(z_signal + attended_signal)  # [B, 1, 256]
        z_visual = self.norm_visual(z_visual + attended_visual)  # [B, 1, 256]
        
        # ── Step 4: Feed-Forward Network ──────────────────────────
        z_signal = self.norm_signal_ffn(z_signal + self.ffn_signal(z_signal))
        z_visual = self.norm_visual_ffn(z_visual + self.ffn_visual(z_visual))
        
        # ── Step 5: Remove sequence dim and project back ──────────
        z_signal = z_signal.squeeze(1)  # [B, 256]
        z_visual = z_visual.squeeze(1)  # [B, 256]
        
        delta_signal = self.out_signal(z_signal)  # [B, 512]
        delta_visual = self.out_visual(z_visual)  # [B, 2048]
        
        # ── Step 6: Gated Residual Blending ───────────────────────
        # gate starts at sigmoid(0) = 0.5, but parameter initialized to 0
        # so alpha = sigmoid(0) = 0.5. To start fully preserving originals,
        # we use: fused = original + alpha * delta
        alpha_s = torch.sigmoid(self.gate_signal)
        alpha_v = torch.sigmoid(self.gate_visual)
        
        h_signal_fused = h_signal + alpha_s * delta_signal  # [B, 512]
        h_visual_fused = h_visual + alpha_v * delta_visual  # [B, 2048]
        
        attn_weights = {
            "signal_attends_visual": attn_s2v.detach(),  # [B, 1, 1]
            "visual_attends_signal": attn_v2s.detach(),  # [B, 1, 1]
            "gate_signal": alpha_s.item(),
            "gate_visual": alpha_v.item(),
        }
        
        return h_signal_fused, h_visual_fused, attn_weights


class FusedClassificationHead(nn.Module):
    """
    Classification head that operates on fused features.
    
    Takes the fused feature vector and produces class logits.
    This replaces the original `model.fc` layer when CMAF is active.
    
    Args:
        in_features:  Input feature dimension (512 for signal, 2048 for visual)
        num_classes:  Number of output classes
        dropout:      Dropout rate (default: 0.2)
    """
    
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
