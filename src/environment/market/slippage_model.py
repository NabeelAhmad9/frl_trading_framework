"""Slippage model — deterministic or stochastic slippage estimation."""

from typing import Any, Dict
import numpy as np


def compute_slippage(
    direction: int,
    pip_size: float,
    config: Dict[str, Any],
    volatility_proxy: float = 0.0,
    session_mult: float = 1.0,
    rng: np.random.RandomState = None,
) -> float:
    """Compute slippage in price terms (always adverse to agent).

    Parameters
    ----------
    direction : int
        1 for buy, -1 for sell.
    pip_size : float
    config : dict
        Slippage config block.
    volatility_proxy : float
    session_mult : float
    rng : np.random.RandomState or None

    Returns
    -------
    float
        Absolute slippage in price terms (always >= 0).
    """
    if not config.get("enabled", True):
        return 0.0

    base_pips = config.get("base_slippage_pips", 0.5)
    vol_mult = config.get("volatility_multiplier", 1.0)
    sess_mult = config.get("session_multiplier", 1.0)
    vol_factor = 1.0 + max(0.0, float(volatility_proxy)) * vol_mult

    mode = config.get("mode", "deterministic")
    if mode == "deterministic":
        slippage_pips = base_pips * vol_factor * sess_mult * session_mult
    else:
        # Stochastic: sample from uniform [0, 2*base_pips]
        if rng is None:
            rng = np.random
        slippage_pips = rng.uniform(0, 2 * base_pips) * vol_factor * sess_mult * session_mult

    return abs(slippage_pips * pip_size)
