import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    The core building block of a ResNet.
    It contains two convolutional layers and a 'skip connection'.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer. This one will handle the downsampling with its stride.
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # The second conv layer always has stride=1. It processes features but doesn't change the length.
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # The 'skip connection' part
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Now the dimensions will match correctly
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    """
    A ResNet-style 1D-CNN for ECG classification.
    """
    def __init__(self, num_classes=5, num_leads=12):
        super(ResNet1D, self).__init__()
        
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# --- Block for testing the model architecture ---
if __name__ == '__main__':
    dummy_input = torch.randn(4, 12, 1000)
    model = ResNet1D(num_classes=5, num_leads=12)
    
    print("--- ResNet Model Architecture ---")
    print(model)
    
    print("\n--- Testing Forward Pass ---")
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Output tensor (logits):")
    print(output)