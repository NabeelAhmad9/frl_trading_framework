"""Volatility penalty — penalize unstable equity returns."""

from typing import Any, Dict, Tuple
from collections import deque
import numpy as np

from src.reward.base_reward import BaseRewardComponent
from src.environment.state.portfolio_state import PortfolioState


class VolatilityPenalty(BaseRewardComponent):

    def __init__(self, weight: float = 0.05, enabled: bool = True, window: int = 20):
        super().__init__(weight, enabled)
        self.returns_buffer = deque(maxlen=window)

    def reset_buffer(self) -> None:
        """Reset internal buffer for new episodes."""
        self.returns_buffer.clear()

    def compute(self, prev_portfolio: PortfolioState, curr_portfolio: PortfolioState,
                action: int, cost_breakdown: Dict[str, float], **kwargs) -> Tuple[float, Dict[str, Any]]:
        # Compute current return to add to buffer
        eps = 1e-8
        ret = (curr_portfolio.equity - prev_portfolio.equity) / max(abs(prev_portfolio.equity), eps)
        self.returns_buffer.append(float(ret))
        
        if len(self.returns_buffer) < 2:
            return 0.0, {"volatility": 0.0}
        
        # Volatility is the std dev of returns in window
        vol = float(np.std(self.returns_buffer))
        # Penaly is positive (subtract from total reward)
        return -vol, {"volatility": vol}
