import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for time-series/sequence data.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class ECGFoundationWrapper(nn.Module):
    """
    Backbone-agnostic wrapper designed to ingest a pre-trained Foundation Model
    (mocked here as a PyTorch TransformerEncoder) and route features to existing
    specialist taxonomy heads.
    """
    def __init__(self, input_channels=12, seq_len=5000, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        
        # 1. Input Projection (Channel Mapping)
        # Maps raw 12-lead ECG to the transformer's expected d_model dimensionality
        self.input_projection = nn.Linear(input_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        
        # 2. Foundation Backbone (Placeholder Transformer)
        # In a real scenario, this would be loaded via a pre-trained checkpoint
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.1, 
            batch_first=True
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Specialist Routing Heads
        self.router_head = nn.Linear(d_model, 2)         # Rhythm vs Structure
        self.structure_head = nn.Linear(d_model, 11)     # 11-class Structure path
        self.rhythm_head = nn.Linear(d_model, 13)        # 13-class Rhythm path

    def freeze_backbone(self):
        """
        Locks the pre-trained Foundation Model base, dropping it entirely from
        the optimizer gradient graph to prevent destructive catastrophic forgetting 
        during fine-tuning.
        """
        for param in self.input_projection.parameters():
            param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor, task: str = 'router') -> torch.Tensor:
        """
        Executes a localized forward pass restricted to the designated classifier.
        
        Args:
            x: Input tensor [B, 12, L]
            task: Specifies which distinct linear head to actuate 
                  ('router', 'structure', 'rhythm')
                  
        Returns:
            Logits constrained to the specified head.
        """
        # Ensure mapping expects [B, L, 12] for standard linear projection
        if x.shape[1] == 12:
            x = x.transpose(1, 2)  # [B, L, 12]
            
        # Project and encode
        x = self.input_projection(x)  # [B, L, d_model]
        x = self.positional_encoding(x)
        
        # Foundation encode
        encoded = self.backbone(x)  # [B, L, d_model]
        
        # Aggregate logic (Global Average Pooling over the sequence length)
        # Replaces [CLS] token extraction for raw multi-lead processing
        pooled_features = encoded.mean(dim=1)  # [B, d_model]
        
        # Specialized Routing
        if task == 'router':
            return self.router_head(pooled_features)
        elif task == 'structure':
            return self.structure_head(pooled_features)
        elif task == 'rhythm':
            return self.rhythm_head(pooled_features)
        else:
            raise ValueError(f"Unknown task request: {task}. Must be router, structure, or rhythm.")
