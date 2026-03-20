"""Legal action mask computation.

Determines valid actions at each step based on portfolio state, market context,
risk constraints, and action-space mode.
"""

import numpy as np
from typing import Any, Dict

from src.environment.action_space import Action, NUM_EXTENDED_ACTIONS, NUM_SIMPLIFIED_ACTIONS
from src.environment.state.portfolio_state import PortfolioState
from src.environment.state.market_state import MarketState
from src.environment.portfolio.risk_constraints import RiskConstraints


def compute_legal_mask(
    portfolio: PortfolioState,
    market: MarketState,
    risk: RiskConstraints,
    simplified_mode: bool = False,
) -> np.ndarray:
    """Compute a binary mask of legal actions.

    Parameters
    ----------
    portfolio : PortfolioState
    market : MarketState
    risk : RiskConstraints
    simplified_mode : bool
        If True, return a 3-element mask for simplified actions.

    Returns
    -------
    np.ndarray
        Binary mask (1=legal, 0=illegal).
    """
    if simplified_mode:
        return _simplified_mask(portfolio, market, risk)

    mask = np.zeros(NUM_EXTENDED_ACTIONS, dtype=np.float32)

    # HOLD is always legal
    mask[Action.HOLD] = 1.0

    is_flat = portfolio.direction == 0
    is_long = portfolio.direction == 1
    is_short = portfolio.direction == -1

    if is_flat:
        # Can open long or short if margin allows
        if risk.can_open_position(portfolio, risk.base_lot_size, market.close, pair=market.pair):
            mask[Action.OPEN_LONG] = 1.0
            mask[Action.OPEN_SHORT] = 1.0
    else:
        # Can reduce, close, or reverse if non-flat
        mask[Action.REDUCE_POSITION] = 1.0
        mask[Action.CLOSE_POSITION] = 1.0
        # Reverse needs margin for new position
        if risk.can_open_position(portfolio, risk.base_lot_size, market.close, pair=market.pair):
            mask[Action.REVERSE_POSITION] = 1.0

        # Pyramid
        if is_long and risk.can_pyramid(portfolio, market.close):
            if risk.can_open_position(portfolio, risk.pyramid_increment, market.close, pair=market.pair):
                mask[Action.PYRAMID_LONG] = 1.0
        if is_short and risk.can_pyramid(portfolio, market.close):
            if risk.can_open_position(portfolio, risk.pyramid_increment, market.close, pair=market.pair):
                mask[Action.PYRAMID_SHORT] = 1.0

        # Martingale
        if is_long and risk.can_martingale(portfolio, market.close):
            add_lots = max(portfolio.total_lots * (risk.martingale_multiplier - 1.0), risk.base_lot_size)
            if risk.can_open_position(portfolio, add_lots, market.close, pair=market.pair):
                mask[Action.MARTINGALE_LONG] = 1.0
        if is_short and risk.can_martingale(portfolio, market.close):
            add_lots = max(portfolio.total_lots * (risk.martingale_multiplier - 1.0), risk.base_lot_size)
            if risk.can_open_position(portfolio, add_lots, market.close, pair=market.pair):
                mask[Action.MARTINGALE_SHORT] = 1.0

    return mask


def _simplified_mask(portfolio: PortfolioState, market: MarketState, risk: RiskConstraints) -> np.ndarray:
    """Simplified 3-action mask.

    HOLD is always legal.  TARGET_LONG / TARGET_SHORT are only legal when the corresponding
    extended action that would be executed is legal (i.e. has sufficient margin).  Previously
    this returned all-ones unconditionally, meaning the agent's Q-value masking provided no
    safety signal about insufficient margin or risk cap violations in simplified mode.
    """
    from src.environment.action_space import SimplifiedAction, Action
    mask = np.zeros(NUM_SIMPLIFIED_ACTIONS, dtype=np.float32)
    mask[SimplifiedAction.HOLD] = 1.0

    is_long = portfolio.direction == 1
    is_short = portfolio.direction == -1
    is_flat = portfolio.direction == 0

    # TARGET_LONG: would map to OPEN_LONG (flat), REVERSE (short), or HOLD (already long)
    if is_long:
        mask[SimplifiedAction.TARGET_LONG] = 1.0   # already long → HOLD, always ok
    elif is_flat:
        if risk.can_open_position(portfolio, risk.base_lot_size, market.close, pair=market.pair):
            mask[SimplifiedAction.TARGET_LONG] = 1.0
    else:  # is_short → REVERSE
        if risk.can_open_position(portfolio, risk.base_lot_size, market.close, pair=market.pair):
            mask[SimplifiedAction.TARGET_LONG] = 1.0

    # TARGET_SHORT: would map to OPEN_SHORT (flat), REVERSE (long), or HOLD (already short)
    if is_short:
        mask[SimplifiedAction.TARGET_SHORT] = 1.0  # already short → HOLD, always ok
    elif is_flat:
        if risk.can_open_position(portfolio, risk.base_lot_size, market.close, pair=market.pair):
            mask[SimplifiedAction.TARGET_SHORT] = 1.0
    else:  # is_long → REVERSE
        if risk.can_open_position(portfolio, risk.base_lot_size, market.close, pair=market.pair):
            mask[SimplifiedAction.TARGET_SHORT] = 1.0

    return mask
