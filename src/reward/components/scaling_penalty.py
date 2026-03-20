"""Scaling penalty — milder for pyramiding, harsher for martingale."""

from typing import Any, Dict, Tuple
from src.reward.base_reward import BaseRewardComponent
from src.environment.state.portfolio_state import PortfolioState
from src.environment.action_space import Action


class ScalingPenalty(BaseRewardComponent):

    def __init__(
        self,
        pyramid_weight: float = 0.05,
        martingale_weight: float = 0.2,
        enabled: bool = True,
        event_scale: float = 0.02,
        max_penalty_per_step: float = 0.02,
    ):
        super().__init__(weight=1.0, enabled=enabled)
        self.pyramid_weight = pyramid_weight
        self.martingale_weight = martingale_weight
        self.event_scale = max(float(event_scale), 0.0)
        self.max_penalty_per_step = max(float(max_penalty_per_step), 0.0)

    def compute(self, prev_portfolio: PortfolioState, curr_portfolio: PortfolioState,
                action: int, cost_breakdown: Dict[str, float], **kwargs) -> Tuple[float, Dict[str, Any]]:
        pyramid_penalty = 0.0
        martingale_penalty = 0.0

        if action in (Action.PYRAMID_LONG, Action.PYRAMID_SHORT):
            raw = -self.event_scale * self.pyramid_weight * max(curr_portfolio.pyramid_levels, 1)
            pyramid_penalty = max(-self.max_penalty_per_step, raw)
        elif action in (Action.MARTINGALE_LONG, Action.MARTINGALE_SHORT):
            raw = -self.event_scale * self.martingale_weight * max(curr_portfolio.martingale_steps, 1)
            martingale_penalty = max(-self.max_penalty_per_step, raw)

        total = pyramid_penalty + martingale_penalty
        return total, {
            "pyramid_penalty": pyramid_penalty,
            "martingale_penalty": martingale_penalty,
            "event_scale": self.event_scale,
            "max_penalty_per_step": self.max_penalty_per_step,
        }
