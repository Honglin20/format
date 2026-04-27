"""Shared toy model for examples — 2-layer MLP with residual connection."""
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
