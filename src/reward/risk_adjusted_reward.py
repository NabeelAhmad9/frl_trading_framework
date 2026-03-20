"""Risk-adjusted reward — assemble final reward from enabled components."""

from typing import Any, Dict, List, Tuple
from collections import OrderedDict

from src.environment.state.portfolio_state import PortfolioState
from src.reward.base_reward import BaseRewardComponent
from src.reward.normalizer import RewardNormalizer


# Deterministic component order
COMPONENT_ORDER = [
    "profit", "holding", "volatility", "drawdown", "transaction",
    "overtrading", "scaling", "margin", "liquidation", "constraint_violation",
]


class RiskAdjustedReward:
    """Assemble final scalar reward from enabled components per config."""

    def __init__(self, components: Dict[str, BaseRewardComponent], normalizer: RewardNormalizer):
        self.components: OrderedDict[str, BaseRewardComponent] = OrderedDict()
        for name in COMPONENT_ORDER:
            if name in components:
                self.components[name] = components[name]
        self.normalizer = normalizer

    def compute(
        self,
        prev_portfolio: PortfolioState,
        curr_portfolio: PortfolioState,
        action: int,
        cost_breakdown: Dict[str, float],
        market_t: Any = None,
        next_market: Any = None,
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute total reward and per-component breakdown."""
        total = 0.0
        breakdown = {}

        for name, comp in self.components.items():
            if not comp.enabled:
                continue
            
            # Use curr_portfolio as truth for components
            raw, diag = comp.compute(
                prev_portfolio, curr_portfolio, action, cost_breakdown, 
                market_t=market_t,
                next_market=next_market,
                **kwargs,
            )
            
            weighted = float(comp.weighted_value(raw))
            total += weighted
            breakdown[name] = {"raw": raw, "weighted": weighted, "diagnostics": diag}

        total_norm = self.normalizer.normalize(total)
        
        # Add summary stats to breakdown
        breakdown["total_unnormalized"] = total
        breakdown["total_normalized"] = total_norm
        # Backward-compatible aliases used by some tests/consumers
        breakdown["total_before_norm"] = total
        breakdown["total_after_norm"] = total_norm
        
        return total_norm, breakdown

    def reset_engine(self) -> None:
        """Resets component buffers for stateful reward components."""
        for name, comp in self.components.items():
            if hasattr(comp, "reset_buffer"):
                comp.reset_buffer()
