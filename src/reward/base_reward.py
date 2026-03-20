"""Base reward component interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from src.environment.state.portfolio_state import PortfolioState


class BaseRewardComponent(ABC):
    """Abstract base for all reward components."""

    def __init__(self, weight: float = 1.0, enabled: bool = True):
        self.weight = weight
        self.enabled = enabled

    @abstractmethod
    def compute(
        self,
        prev_portfolio: PortfolioState,
        curr_portfolio: PortfolioState,
        action: int,
        cost_breakdown: Dict[str, float],
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute scalar component contribution and diagnostics.

        Returns
        -------
        Tuple of (raw_value, diagnostics_dict).
        """

    def weighted_value(self, raw: float) -> float:
        return raw * self.weight if self.enabled else 0.0
