"""Model factory — build encoder + head composition from config."""

import torch
import torch.nn as nn
from typing import Any, Dict

from src.models.encoders.mlp_encoder import MLPEncoder
from src.models.encoders.cnn_encoder import CNNEncoder
from src.models.heads.q_head import QHead
from src.models.heads.dueling_head import DuelingHead


class QNetwork(nn.Module):
    """Composed Q-network: encoder + head."""

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


class FlattenedCNNEncoder(nn.Module):
    """CNN encoder adapter for flattened observations.

    Input contract:
    - first ``window_length * num_features`` values are market-window features,
    - remaining values are auxiliary features (portfolio, legal mask, etc.).
    """

    def __init__(
        self,
        num_features: int,
        window_length: int,
        extra_dim: int,
        hidden_dims,
        activation: str = "relu",
        dropout: float = 0.0,
        num_filters: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.window_length = int(window_length)
        self.extra_dim = int(max(extra_dim, 0))
        self.market_dim = int(self.num_features * self.window_length)

        self.cnn = CNNEncoder(
            num_features=self.num_features,
            window_length=self.window_length,
            num_filters=int(num_filters),
            kernel_size=int(kernel_size),
        )

        trunk_input_dim = int(self.cnn.output_dim + self.extra_dim)
        if hidden_dims:
            self.trunk = MLPEncoder(trunk_input_dim, hidden_dims, activation, dropout)
            self.output_dim = self.trunk.output_dim
        else:
            self.trunk = nn.Identity()
            self.output_dim = trunk_input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        market_flat = x[:, : self.market_dim]
        market = market_flat.view(x.shape[0], self.window_length, self.num_features)
        market_features = self.cnn(market)

        if self.extra_dim > 0:
            extra = x[:, self.market_dim : self.market_dim + self.extra_dim]
            features = torch.cat([market_features, extra], dim=1)
        else:
            features = market_features

        return self.trunk(features)


def build_model(input_dim: int, num_actions: int, config: Dict[str, Any]) -> QNetwork:
    """Build QNetwork from agent config.

    Parameters
    ----------
    input_dim : int
        Flattened observation dimension.
    num_actions : int
    config : dict
        Agent config block.

    Returns
    -------
    QNetwork
    """
    agent_cfg = config.get("agent", config)
    model_cfg = agent_cfg.get("model", {})

    encoder_type = model_cfg.get("encoder_type", "mlp")
    hidden_dims = model_cfg.get("hidden_dims", [256, 128])
    dueling = model_cfg.get("dueling", False)
    activation = model_cfg.get("activation", "relu")
    dropout = model_cfg.get("dropout", 0.0)

    if encoder_type == "mlp":
        encoder = MLPEncoder(input_dim, hidden_dims, activation, dropout)
    elif encoder_type == "cnn":
        env_cfg = config.get("environment", {})
        obs_cfg = env_cfg.get("observation", {})
        window_length = int(obs_cfg.get("window_length", 1))

        # Flat observation layout from observation_builder:
        # market_flat + portfolio(10) + legal_mask(num_actions)
        portfolio_dim = 10
        mask_dim = int(num_actions)
        extra_dim = int(portfolio_dim + mask_dim)
        market_dim = int(max(input_dim - extra_dim, 0))

        if window_length <= 0 or market_dim <= 0 or (market_dim % window_length) != 0:
            # Safe fallback for malformed shapes.
            encoder = MLPEncoder(input_dim, hidden_dims, activation, dropout)
        else:
            num_features = int(market_dim // window_length)
            cnn_cfg = model_cfg.get("cnn", {})
            encoder = FlattenedCNNEncoder(
                num_features=num_features,
                window_length=window_length,
                extra_dim=extra_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
                num_filters=cnn_cfg.get("num_filters", 32),
                kernel_size=cnn_cfg.get("kernel_size", 3),
            )
    else:
        encoder = MLPEncoder(input_dim, hidden_dims, activation, dropout)

    if dueling:
        head = DuelingHead(encoder.output_dim, num_actions)
    else:
        head = QHead(encoder.output_dim, num_actions)

    model = QNetwork(encoder, head)
    param_count = sum(p.numel() for p in model.parameters())
    return model
