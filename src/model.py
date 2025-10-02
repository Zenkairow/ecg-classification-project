import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DCNN(nn.Module):
    """
    A simple 1D Convolutional Neural Network for ECG classification.
    """
    def __init__(self, num_classes=5, num_leads=12, signal_length=1000):
        """
        Initializes the model architecture.

        Args:
            num_classes (int): The number of output classes to predict.
            num_leads (int): The number of ECG leads (channels).
            signal_length (int): The length of the input signal sequence.
        """
        super(Simple1DCNN, self).__init__()
        
        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv1d(in_channels=num_leads, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # --- Convolutional Block 3 ---
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # --- Fully Connected Layers ---
        # Calculate the flattened size after pooling
        flattened_size = 128 * (signal_length // 8)
        
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, num_leads, signal_length].
        
        Returns:
            torch.Tensor: The output logits of shape [batch_size, num_classes].
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# --- Block for testing the model architecture ---
if __name__ == '__main__':
    # Create a dummy input tensor to test the model
    dummy_input = torch.randn(4, 12, 1000)
    
    # Create an instance of the model
    model = Simple1DCNN(num_classes=5, num_leads=12, signal_length=1000)
    
    print("--- Model Architecture ---")
    print(model)
    
    print("\n--- Testing Forward Pass ---")
    # Pass the dummy input through the model
    output = model(dummy_input)
    
    # Print the shape of the input and output
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Output tensor (logits):")
    print(output)