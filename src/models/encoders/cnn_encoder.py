"""CNN encoder — process windowed feature observations (optional)."""

import torch
import torch.nn as nn
from typing import List


class CNNEncoder(nn.Module):
    """1D Conv encoder for windowed market observations."""

    def __init__(self, num_features: int, window_length: int, num_filters: int = 32, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, num_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.output_dim = num_filters * window_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window, features] -> [batch, features, window]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x.flatten(1)
