"""Sharpe reward — optional short-horizon risk-adjusted bonus (disabled by default)."""

from typing import Any, Dict, Tuple
from collections import deque
import numpy as np

from src.reward.base_reward import BaseRewardComponent
from src.environment.state.portfolio_state import PortfolioState


class SharpeReward(BaseRewardComponent):

    def __init__(self, weight: float = 0.0, enabled: bool = False, window: int = 20):
        super().__init__(weight, enabled)
        self.returns_buffer = deque(maxlen=window)

    def reset_buffer(self) -> None:
        """Reset internal buffer for new episodes."""
        self.returns_buffer.clear()

    def compute(self, prev_portfolio: PortfolioState, curr_portfolio: PortfolioState,
                action: int, cost_breakdown: Dict[str, float], **kwargs) -> Tuple[float, Dict[str, Any]]:
        # Reward computation depends ONLY on current and previous portfolio
        # To maintain reproducibility across parallel workers, ensure no side effects
        # unless strictly managed by the environment reset.
        eps = 1e-8
        ret = (curr_portfolio.equity - prev_portfolio.equity) / max(abs(prev_portfolio.equity), eps)
        self.returns_buffer.append(float(ret))
        
        if len(self.returns_buffer) < 2:
            return 0.0, {"sharpe": 0.0}
        
        arr = np.array(self.returns_buffer)
        mean_ret = arr.mean()
        std_ret = arr.std()
        
        if std_ret < eps:
            return 0.0, {"sharpe": 0.0, "mean": mean_ret, "std": std_ret}
            
        sharpe = float(mean_ret / std_ret)
        return sharpe, {"sharpe": sharpe, "mean": mean_ret, "std": std_ret}
