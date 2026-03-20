"""Liquidation penalty — severe penalty on forced liquidation."""

from typing import Any, Dict, Tuple
from src.reward.base_reward import BaseRewardComponent
from src.environment.state.portfolio_state import PortfolioState


class LiquidationPenalty(BaseRewardComponent):

    def compute(self, prev_portfolio: PortfolioState, curr_portfolio: PortfolioState,
                action: int, cost_breakdown: Dict[str, float], **kwargs) -> Tuple[float, Dict[str, Any]]:
        if curr_portfolio.forced_liquidations > prev_portfolio.forced_liquidations:
            return -1.0, {"liquidation": True}
        return 0.0, {"liquidation": False}
