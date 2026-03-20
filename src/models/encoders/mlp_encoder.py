"""MLP encoder — configurable dense stack."""

import torch
import torch.nn as nn
from typing import List


class MLPEncoder(nn.Module):
    """Dense encoder with configurable hidden layers."""

    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}.get(activation, nn.ReLU)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
