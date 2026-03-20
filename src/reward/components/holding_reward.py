"""Holding reward — small positive reward for profitable, controlled-risk positions."""

from typing import Any, Dict, Tuple
from src.reward.base_reward import BaseRewardComponent
from src.environment.state.portfolio_state import PortfolioState


class HoldingReward(BaseRewardComponent):

    def compute(self, prev_portfolio: PortfolioState, curr_portfolio: PortfolioState,
                action: int, cost_breakdown: Dict[str, float], **kwargs) -> Tuple[float, Dict[str, Any]]:
        # Reward for holding a profitable position with controlled risk
        if curr_portfolio.direction == 0:
            return 0.0, {"holding": 0.0}
            
        # Reward if unrealized PnL is positive AND current drawdown is low
        # This keeps the agent from closing profitable positions too early
        # but also not holding losers.
        
        # Use relative unrealized PnL%
        unrealized_pct = curr_portfolio.unrealized_pnl / max(abs(curr_portfolio.equity), 1e-8)
        
        # Binary reward criteria: profitable AND low DD
        if unrealized_pct > 0.001 and curr_portfolio.current_drawdown < 0.02:
            reward = 0.1 # Small bonus
        else:
            reward = 0.0
            
        return float(reward), {"holding": reward, "unrealized_pct": unrealized_pct}
