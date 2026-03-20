"""Constraint violation penalty — penalize illegal action attempts."""

from typing import Any, Dict, Tuple

from src.reward.base_reward import BaseRewardComponent
from src.environment.state.portfolio_state import PortfolioState


class ConstraintViolationPenalty(BaseRewardComponent):
    """Penalize illegal-action violations surfaced by the environment."""

    def __init__(self, weight: float = 0.1, enabled: bool = True, base_penalty: float = 0.02):
        super().__init__(weight=weight, enabled=enabled)
        self.base_penalty = max(float(base_penalty), 0.0)

    def compute(
        self,
        prev_portfolio: PortfolioState,
        curr_portfolio: PortfolioState,
        action: int,
        cost_breakdown: Dict[str, float],
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        was_legal = bool(kwargs.get("was_legal", True))
        invalid_action_penalty = float(kwargs.get("invalid_action_penalty", 0.0) or 0.0)

        if was_legal:
            return 0.0, {"constraint_violation": False, "penalty": 0.0}

        penalty_mag = max(self.base_penalty, invalid_action_penalty)
        penalty = -penalty_mag
        return penalty, {
            "constraint_violation": True,
            "base_penalty": self.base_penalty,
            "invalid_action_penalty": invalid_action_penalty,
            "penalty": penalty,
        }
