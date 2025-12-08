import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create constant positional encoding matrix with values in range [0, max_len)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Debugging Shapes
        # print(f"DEBUG PE: x={x.shape}, pe={self.pe.shape}")
        return x + self.pe[:, :x.size(1), :]

class ECGTransformer(nn.Module):
    def __init__(self, num_classes=50, input_channels=12, d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1):
        super(ECGTransformer, self).__init__()
        
        # 1. Feature Projection (12 Leads -> d_model)
        # We treat the 1000 time steps as the sequence length.
        # Input shape expected: [Batch, 12, 1000] -> Permute to [Batch, 1000, 12]
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classification Head
        # Global Average Pooling then Linear
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input: [Batch, 12, 1000]
        # Transpose to [Batch, 1000, 12] for Linear layer (channels last)
        x = x.permute(0, 2, 1) 
        
        # Project to d_model
        x = self.input_projection(x) # [Batch, 1000, d_model]
        
        # Add Positional Encoding
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        x = self.transformer_encoder(x) # [Batch, 1000, d_model]
        
        # Global Average Pooling (over time dimension)
        x = x.mean(dim=1) # [Batch, d_model]
        
        # Classifier
        x = self.fc(x)
        
        return x

if __name__ == "__main__":
    # Test Block
    dummy_input = torch.randn(8, 12, 1000) # Batch 8, 12 leads, 1000 samples
    model = ECGTransformer(num_classes=50)
    output = model(dummy_input)
    print(f"Model V6 Test Input: {dummy_input.shape}")
    print(f"Model V6 Test Output: {output.shape}") # Should be [8, 50]
