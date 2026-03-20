"""Observation builder — construct stable observation dicts for agents."""

import numpy as np
from typing import Any, Dict, List

from src.environment.state.portfolio_state import PortfolioState
from src.environment.state.market_state import MarketState


def build_observation(
    feature_matrix: np.ndarray,
    portfolio: PortfolioState,
    legal_mask: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Build canonical observation dictionary.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Shape [window_length, num_features] or [num_features].
    portfolio : PortfolioState
    legal_mask : np.ndarray
        Shape [num_actions].
    config : dict

    Returns
    -------
    dict with keys: 'market', 'portfolio', 'mask', optionally 'flat'.
    """
    env_cfg = config.get("environment", config)
    obs_cfg = env_cfg.get("observation", {})
    initial_capital = env_cfg.get("account", {}).get("initial_capital", 100000.0)

    # Market features
    if feature_matrix.ndim == 1:
        market_obs = feature_matrix.reshape(1, -1).astype(np.float32)
    else:
        market_obs = feature_matrix.astype(np.float32)

    # Portfolio vector (normalized)
    eps = 1e-8
    cap = max(initial_capital, eps)
    portfolio_vec = np.array([
        portfolio.cash / cap,
        portfolio.equity / cap,
        portfolio.unrealized_pnl / cap,
        portfolio.used_margin / cap,
        portfolio.free_margin / cap,
        float(portfolio.direction),
        portfolio.total_lots,
        float(portfolio.pyramid_levels),
        float(portfolio.martingale_steps),
        portfolio.current_drawdown,
    ], dtype=np.float32)

    obs = {
        "market": market_obs,
        "portfolio": portfolio_vec,
        "mask": legal_mask.astype(np.float32),
    }

    # Flattened for MLP consumption
    flat = np.concatenate([market_obs.flatten(), portfolio_vec, legal_mask])
    obs["flat"] = flat.astype(np.float32)

    return obs
