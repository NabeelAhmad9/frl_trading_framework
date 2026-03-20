"""Drawdown penalty — penalize current or incremental drawdown growth."""

from typing import Any, Dict, Tuple
from src.reward.base_reward import BaseRewardComponent
from src.environment.state.portfolio_state import PortfolioState


class DrawdownPenalty(BaseRewardComponent):

    def compute(self, prev_portfolio: PortfolioState, curr_portfolio: PortfolioState,
                action: int, cost_breakdown: Dict[str, float], **kwargs) -> Tuple[float, Dict[str, Any]]:
        # Penalize incremental growth in drawdown or severe depth
        current_dd = curr_portfolio.current_drawdown
        previous_dd = prev_portfolio.current_drawdown
        
        # Incremental drawdown (change in DD)
        # Only penalize if DD increased
        dd_increase = max(0.0, current_dd - previous_dd)
        
        # Heavy penalty for hitting critical levels
        critical_penalty = 0.0
        if current_dd > 0.15: # Critical threshold
            critical_penalty = (current_dd - 0.15) * 5.0
            
        total_penalty = -(dd_increase + critical_penalty)
        return float(total_penalty), {
            "current_drawdown": current_dd, 
            "dd_increase": dd_increase,
            "critical_penalty": critical_penalty
        }
