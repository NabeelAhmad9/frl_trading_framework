"""Standard Q-value head."""

import torch
import torch.nn as nn


class QHead(nn.Module):
    """Output Q-values for each action."""

    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.q = nn.Linear(input_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q(x)
