"""PnL reward — gross per-step equity return before explicit transaction penalties."""

from typing import Any, Dict, Tuple
from src.reward.base_reward import BaseRewardComponent
from src.environment.state.portfolio_state import PortfolioState


class PnlReward(BaseRewardComponent):

    def __init__(self, weight: float = 1.0, enabled: bool = True, return_scale: float = 1.0):
        super().__init__(weight=weight, enabled=enabled)
        self.return_scale = max(float(return_scale), 1e-8)

    def compute(self, prev_portfolio: PortfolioState, curr_portfolio: PortfolioState,
                action: int, cost_breakdown: Dict[str, float], **kwargs) -> Tuple[float, Dict[str, Any]]:
        eps = 1e-8
        # Use starting equity of the period to normalize return.
        # We compute a gross (pre-explicit-cost-penalty) return to avoid double-counting
        # transaction costs when a separate transaction component is enabled.
        #
        # net_delta = equity_after - equity_before
        # gross_delta ~= net_delta + total_cost
        #
        # total_cost here is sourced from environment execution accounting for this step.
        equity_before = max(abs(prev_portfolio.equity), eps)
        net_delta = curr_portfolio.equity - prev_portfolio.equity
        total_cost = float(cost_breakdown.get("total_cost", 0.0))
        gross_delta = net_delta + total_cost
        gross_return = gross_delta / equity_before
        scaled_return = gross_return * self.return_scale
        
        # Ensure result is a float, not a numpy type
        net_delta = float(net_delta)
        gross_return = float(gross_return)
        scaled_return = float(scaled_return)
        return scaled_return, {
            "net_delta": net_delta,
            "gross_return": gross_return,
            "total_cost_added_back": total_cost,
            "return_scale": self.return_scale,
            "scaled_return": scaled_return,
        }
