"""Margin penalty — penalize dangerous margin utilization before liquidation."""

from typing import Any, Dict, Tuple
from src.reward.base_reward import BaseRewardComponent
from src.environment.state.portfolio_state import PortfolioState


class MarginPenalty(BaseRewardComponent):

    def compute(self, prev_portfolio: PortfolioState, curr_portfolio: PortfolioState,
                action: int, cost_breakdown: Dict[str, float], **kwargs) -> Tuple[float, Dict[str, Any]]:
        # Equity must be positive for valid utilization check
        if curr_portfolio.equity <= 0:
            return -1.0, {"margin_util": 0.0, "penalty": -1.0}
            
        if curr_portfolio.used_margin <= 0:
            return 0.0, {"margin_util": 0.0, "penalty": 0.0}
            
        margin_util = curr_portfolio.used_margin / curr_portfolio.equity
        # Thresholded penalty: only penalize when utilization is above a safe limit (e.g. 1.0)
        # Assuming used_margin is not scaled by leverage already
        if margin_util > 5.0: # High leverage above 5x
            # Quadratic penalty for extreme leverage
            penalty = -((margin_util - 5.0) ** 2)
        else:
            penalty = 0.0
            
        return float(penalty), {"margin_util": margin_util, "penalty": penalty}
