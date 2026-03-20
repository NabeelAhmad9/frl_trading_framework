"""Transaction penalty — penalize spread, slippage, commission, rollover burden."""

from typing import Any, Dict, Tuple
from src.reward.base_reward import BaseRewardComponent
from src.environment.state.portfolio_state import PortfolioState


class TransactionPenalty(BaseRewardComponent):

    def compute(self, prev_portfolio: PortfolioState, curr_portfolio: PortfolioState,
                action: int, cost_breakdown: Dict[str, float], **kwargs) -> Tuple[float, Dict[str, Any]]:
        total_cost = cost_breakdown.get("total_cost", 0.0)
        eps = 1e-8
        cap = max(abs(prev_portfolio.equity), eps)
        penalty = -total_cost / cap
        return penalty, {"transaction_cost": total_cost, "penalty": penalty}
