"""
V3.0 AdvancedCardiacNet — Multi-Scale Inception-Residual Network with CBAM Attention
=====================================================================================
Drop-in replacement for SEResNet34. Same constructor signature and I/O shape.
    Constructor: AdvancedCardiacNet(num_classes, input_channels)
    Forward:     [B, input_channels, seq_len] -> [B, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CBAM (Convolutional Block Attention Module) — 1D variant
# =============================================================================

class ChannelAttention(nn.Module):
    """Recalibrates lead (channel) importance via squeeze-excitation."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """Highlights specific timing intervals via spatial attention."""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(combined))
        return x * attention


class CBAM(nn.Module):
    """Sequential Channel -> Spatial attention."""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# =============================================================================
# Multi-Scale Inception-Residual Block
# =============================================================================

class InceptionResidualBlock(nn.Module):
    """
    Parallel 3x1, 5x1, 7x1 convolutions -> concat -> 1x1 projection -> CBAM -> residual add.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        # Each branch outputs out_channels // 3 (handle remainder in branch_c)
        branch_ch = out_channels // 3
        branch_c_ch = out_channels - 2 * branch_ch  # absorbs remainder

        # Branch A: 3x1
        self.branch_a = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(branch_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(branch_ch, branch_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(branch_ch),
        )

        # Branch B: 5x1
        self.branch_b = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.BatchNorm1d(branch_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(branch_ch, branch_ch, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(branch_ch),
        )

        # Branch C: 7x1
        self.branch_c = nn.Sequential(
            nn.Conv1d(in_channels, branch_c_ch, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(branch_c_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(branch_c_ch, branch_c_ch, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(branch_c_ch),
        )

        # CBAM after concatenation
        self.cbam = CBAM(out_channels)

        # Residual connection
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)

        out = torch.cat([a, b, c], dim=1)  # [B, out_channels, L]
        out = self.cbam(out)
        out = self.dropout(out)

        out += identity
        out = self.relu(out)
        return out


# =============================================================================
# AdvancedCardiacNet — Full Network
# =============================================================================

class AdvancedCardiacNet(nn.Module):
    """
    Multi-Scale Inception-Residual Network with CBAM for 1D ECG classification.
    Drop-in replacement for SEResNet34.
    
    Args:
        num_classes (int): Number of output classes (default: 25).
        input_channels (int): Number of ECG leads (default: 12).
        layers (list): Number of blocks per stage, ResNet-34 config [3,4,6,3].
    """
    def __init__(self, num_classes=25, input_channels=12, layers=None):
        super().__init__()
        if layers is None:
            layers = [3, 4, 6, 3]  # ResNet-34 config

        self.in_channels = 64

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Stages
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        # Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

        # Weight initialization
        self._init_weights()

    def _make_layer(self, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(InceptionResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(InceptionResidualBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
