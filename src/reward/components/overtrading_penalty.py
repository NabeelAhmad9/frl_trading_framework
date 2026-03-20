"""Overtrading penalty — penalize unnecessary turnover and churn."""

from typing import Any, Dict, Tuple
from collections import deque

from src.reward.base_reward import BaseRewardComponent
from src.environment.state.portfolio_state import PortfolioState
from src.environment.action_space import Action


class OvertradingPenalty(BaseRewardComponent):

    def __init__(
        self,
        weight: float = 0.05,
        enabled: bool = True,
        window: int = 20,
        threshold: int = 10,
        max_penalty: float = 0.02,
    ):
        super().__init__(weight, enabled)
        self.action_buffer = deque(maxlen=window)
        self.threshold = max(int(threshold), 0)
        self.max_penalty = max(float(max_penalty), 0.0)

    def reset_buffer(self) -> None:
        """Reset internal action-history state on episode reset."""
        self.action_buffer.clear()

    def compute(self, prev_portfolio: PortfolioState, curr_portfolio: PortfolioState,
                action: int, cost_breakdown: Dict[str, float], **kwargs) -> Tuple[float, Dict[str, Any]]:
        is_trade = action != Action.HOLD
        self.action_buffer.append(1 if is_trade else 0)
        trade_count = sum(self.action_buffer)

        window_len = max(len(self.action_buffer), 1)
        trade_ratio = trade_count / window_len
        threshold_ratio = min(self.threshold / max(self.action_buffer.maxlen or 1, 1), 1.0)

        if trade_ratio > threshold_ratio and threshold_ratio < 1.0:
            # Smooth bounded penalty in [0, -max_penalty]
            excess_ratio = (trade_ratio - threshold_ratio) / max(1.0 - threshold_ratio, 1e-8)
            penalty = -self.max_penalty * (excess_ratio ** 2)
        else:
            penalty = 0.0

        return float(penalty), {
            "trade_count_window": trade_count,
            "trade_ratio_window": trade_ratio,
            "threshold_ratio": threshold_ratio,
            "max_penalty": self.max_penalty,
            "penalty": float(penalty),
        }
