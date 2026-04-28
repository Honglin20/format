"""Shared toy models for examples."""
import torch.nn as nn
import torch.nn.functional as F


class ToyMLP(nn.Module):
    """Simple MLP suitable for testing all quantization features.

    Contains: Linear, GELU, LayerNorm, residual add — enough to exercise
    module replacement AND inline-op patching (F.gelu, +) in quantize_model.
    """

    def __init__(self, hidden_size: int = 128, intermediate_size: int = 512):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.fc1(x)
        x = F.gelu(x)  # inline op
        x = self.fc2(x)
        x = x + residual  # inline op
        return x


class ToyConvNet(nn.Module):
    """Small CNN for testing Conv2d channel-axis quantization (channel_axis=1).

    Accepts the same ``(B, 128)`` input as :class:`ToyMLP` by reshaping
    internally to ``(B, 1, 16, 8)``.  Contains Conv2d, BatchNorm2d, ReLU,
    and a Linear head — exercises both Conv2d (channel_axis=1 in NCHW) and
    Linear (channel_axis=-1) quantization paths.
    """

    def __init__(self, hidden_channels: int = 16, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels // 2)
        self.conv2 = nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d((4, 2))
        self.fc = nn.Linear(hidden_channels * 4 * 2, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), 1, 16, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)
