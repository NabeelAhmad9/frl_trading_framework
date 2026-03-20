"""Dueling Q-value head — value + advantage streams."""

import torch
import torch.nn as nn


class DuelingHead(nn.Module):
    """Dueling architecture: V(s) + A(s,a) - mean(A)."""

    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.value = nn.Linear(input_dim, 1)
        self.advantage = nn.Linear(input_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.value(x)
        a = self.advantage(x)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q
