"""Reward factory — construct reward engine from resolved config."""

from typing import Any, Dict

from src.reward.base_reward import BaseRewardComponent
from src.reward.normalizer import RewardNormalizer
from src.reward.risk_adjusted_reward import RiskAdjustedReward
from src.reward.components.pnl_reward import PnlReward
from src.reward.components.sharpe_reward import SharpeReward
from src.reward.components.drawdown_penalty import DrawdownPenalty
from src.reward.components.volatility_penalty import VolatilityPenalty
from src.reward.components.transaction_penalty import TransactionPenalty
from src.reward.components.holding_reward import HoldingReward
from src.reward.components.scaling_penalty import ScalingPenalty
from src.reward.components.liquidation_penalty import LiquidationPenalty
from src.reward.components.margin_penalty import MarginPenalty
from src.reward.components.overtrading_penalty import OvertradingPenalty
from src.reward.components.constraint_violation_penalty import ConstraintViolationPenalty


def build_reward_engine(config: Dict[str, Any]) -> RiskAdjustedReward:
    """Build reward engine from resolved config."""
    reward_cfg = config.get("reward", {})
    comp_cfg = reward_cfg.get("components", {})
    weights = reward_cfg.get("weights", {})
    overtrading_cfg = reward_cfg.get("overtrading", {})
    scaling_cfg = reward_cfg.get("scaling", {})
    violation_cfg = reward_cfg.get("constraint_violation", {})
    profit_cfg = comp_cfg.get("profit", {})

    components: Dict[str, BaseRewardComponent] = {}

    # Profit
    if comp_cfg.get("profit", {}).get("enabled", True):
        components["profit"] = PnlReward(
            weight=weights.get("profit", 1.0),
            return_scale=profit_cfg.get("return_scale", 1.0),
        )

    # Holding
    if comp_cfg.get("holding", {}).get("enabled", True):
        components["holding"] = HoldingReward(weight=weights.get("holding", 0.02))

    # Volatility
    if comp_cfg.get("volatility", {}).get("enabled", True):
        components["volatility"] = VolatilityPenalty(weight=weights.get("volatility", 0.05))

    # Drawdown
    if comp_cfg.get("drawdown", {}).get("enabled", True):
        components["drawdown"] = DrawdownPenalty(weight=weights.get("drawdown", 0.2))

    # Transaction
    if comp_cfg.get("transaction", {}).get("enabled", True):
        components["transaction"] = TransactionPenalty(weight=weights.get("transaction", 1.0))

    # Overtrading
    if comp_cfg.get("overtrading", {}).get("enabled", True):
        window = overtrading_cfg.get("window", 20)
        threshold = overtrading_cfg.get("threshold", 10)
        max_penalty = overtrading_cfg.get("max_penalty", 0.02)
        components["overtrading"] = OvertradingPenalty(
            weight=weights.get("overtrading", 0.05),
            window=window,
            threshold=threshold,
            max_penalty=max_penalty,
        )

    # Scaling
    if comp_cfg.get("scaling", {}).get("enabled", True):
        components["scaling"] = ScalingPenalty(
            pyramid_weight=weights.get("pyramid_penalty", 0.05),
            martingale_weight=weights.get("martingale_penalty", 0.2),
            event_scale=scaling_cfg.get("event_scale", 0.02),
            max_penalty_per_step=scaling_cfg.get("max_penalty_per_step", 0.02),
        )

    # Margin
    if comp_cfg.get("margin", {}).get("enabled", True):
        components["margin"] = MarginPenalty(weight=weights.get("margin", 0.1))

    # Liquidation
    if comp_cfg.get("liquidation", {}).get("enabled", True):
        components["liquidation"] = LiquidationPenalty(weight=weights.get("liquidation", 5.0))

    # Constraint violation
    if comp_cfg.get("constraint_violation", {}).get("enabled", False):
        components["constraint_violation"] = ConstraintViolationPenalty(
            weight=weights.get("constraint_violation", 0.1),
            base_penalty=violation_cfg.get("base_penalty", 0.02),
        )

    # Sharpe (disabled by default)
    if comp_cfg.get("sharpe", {}).get("enabled", False):
        components["sharpe"] = SharpeReward(weight=weights.get("sharpe", 0.0))

    normalizer = RewardNormalizer(config)
    return RiskAdjustedReward(components, normalizer)
