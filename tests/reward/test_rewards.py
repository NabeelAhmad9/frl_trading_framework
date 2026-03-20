"""Tests for reward components, aggregation, and normalization."""

import pytest
from src.environment.state.portfolio_state import PortfolioState
from src.environment.action_space import Action
from src.reward.reward_factory import build_reward_engine
from src.reward.components.pnl_reward import PnlReward
from src.reward.components.drawdown_penalty import DrawdownPenalty
from src.reward.components.transaction_penalty import TransactionPenalty
from src.reward.components.scaling_penalty import ScalingPenalty
from src.reward.components.liquidation_penalty import LiquidationPenalty
from src.reward.components.holding_reward import HoldingReward
from src.reward.normalizer import RewardNormalizer


def _default_config():
    return {
        "reward": {
            "components": {
                "profit": {"enabled": True},
                "holding": {"enabled": True},
                "volatility": {"enabled": True},
                "drawdown": {"enabled": True},
                "transaction": {"enabled": True},
                "overtrading": {"enabled": True},
                "scaling": {"enabled": True},
                "margin": {"enabled": True},
                "liquidation": {"enabled": True},
                "constraint_violation": {"enabled": False},
                "sharpe": {"enabled": False},
            },
            "weights": {
                "profit": 1.0,
                "holding": 0.02,
                "volatility": 0.05,
                "drawdown": 0.2,
                "transaction": 1.0,
                "overtrading": 0.05,
                "pyramid_penalty": 0.05,
                "martingale_penalty": 0.2,
                "margin": 0.1,
                "liquidation": 5.0,
                "constraint_violation": 0.5,
            },
            "normalization": {"method": "clip_only", "clip_min": -10.0, "clip_max": 10.0},
            "overtrading": {"window": 20, "threshold": 10},
        }
    }


class TestPnlRewardSign:
    def test_positive_equity_returns_positive(self):
        comp = PnlReward(weight=1.0)
        prev = PortfolioState.initial()
        curr = prev.copy()
        curr.equity = 101000
        raw, _ = comp.compute(prev, curr, Action.HOLD, {})
        assert raw > 0

    def test_negative_equity_returns_negative(self):
        comp = PnlReward(weight=1.0)
        prev = PortfolioState.initial()
        curr = prev.copy()
        curr.equity = 99000
        raw, _ = comp.compute(prev, curr, Action.HOLD, {})
        assert raw < 0


class TestDrawdownPenalty:
    def test_drawdown_negative_penalty(self):
        comp = DrawdownPenalty(weight=0.2)
        prev = PortfolioState.initial()
        curr = prev.copy()
        curr.current_drawdown = 0.1
        raw, _ = comp.compute(prev, curr, Action.HOLD, {})
        assert raw < 0

    def test_zero_drawdown_zero_penalty(self):
        comp = DrawdownPenalty(weight=0.2)
        prev = PortfolioState.initial()
        curr = prev.copy()
        raw, _ = comp.compute(prev, curr, Action.HOLD, {})
        assert raw == 0.0


class TestTransactionPenalty:
    def test_cost_produces_penalty(self):
        comp = TransactionPenalty(weight=1.0)
        prev = PortfolioState.initial()
        curr = prev.copy()
        raw, _ = comp.compute(prev, curr, Action.OPEN_LONG, {"total_cost": 50.0})
        assert raw < 0

    def test_no_cost_no_penalty(self):
        comp = TransactionPenalty(weight=1.0)
        prev = PortfolioState.initial()
        curr = prev.copy()
        raw, _ = comp.compute(prev, curr, Action.HOLD, {"total_cost": 0.0})
        assert raw == 0.0


class TestScalingPenaltyAsymmetry:
    """Martingale penalty must be harsher than pyramid penalty for equivalent depth."""

    def test_martingale_harsher_than_pyramid(self):
        comp = ScalingPenalty(pyramid_weight=0.05, martingale_weight=0.2)
        prev = PortfolioState.initial()

        curr_pyr = prev.copy()
        curr_pyr.pyramid_levels = 1
        raw_pyr, _ = comp.compute(prev, curr_pyr, Action.PYRAMID_LONG, {})

        curr_mart = prev.copy()
        curr_mart.martingale_steps = 1
        raw_mart, _ = comp.compute(prev, curr_mart, Action.MARTINGALE_LONG, {})

        assert abs(raw_mart) > abs(raw_pyr)


class TestLiquidationPenalty:
    def test_liquidation_event_penalty(self):
        comp = LiquidationPenalty(weight=5.0)
        prev = PortfolioState.initial()
        curr = prev.copy()
        curr.forced_liquidations = 1
        raw, _ = comp.compute(prev, curr, Action.HOLD, {})
        assert raw < 0

    def test_no_liquidation_no_penalty(self):
        comp = LiquidationPenalty(weight=5.0)
        prev = PortfolioState.initial()
        curr = prev.copy()
        raw, _ = comp.compute(prev, curr, Action.HOLD, {})
        assert raw == 0.0


class TestHoldingReward:
    def test_profitable_position_rewarded(self):
        comp = HoldingReward(weight=0.02)
        prev = PortfolioState.initial()
        curr = prev.copy()
        curr.direction = 1
        curr.unrealized_pnl = 500
        curr.current_drawdown = 0.01
        raw, _ = comp.compute(prev, curr, Action.HOLD, {})
        assert raw > 0

    def test_flat_no_reward(self):
        comp = HoldingReward(weight=0.02)
        prev = PortfolioState.initial()
        curr = prev.copy()
        raw, _ = comp.compute(prev, curr, Action.HOLD, {})
        assert raw == 0.0


class TestNormalizer:
    def test_clip_within_bounds(self):
        norm = RewardNormalizer({"reward": {"normalization": {"method": "clip_only", "clip_min": -5.0, "clip_max": 5.0}}})
        assert norm.normalize(3.0) == 3.0
        assert norm.normalize(10.0) == 5.0
        assert norm.normalize(-10.0) == -5.0


class TestDecompositionSum:
    def test_decomposition_consistency(self):
        config = _default_config()
        engine = build_reward_engine(config)
        prev = PortfolioState.initial()
        curr = prev.copy()
        curr.equity = 100500
        curr.unrealized_pnl = 500

        total, breakdown = engine.compute(prev, curr, Action.HOLD, {"total_cost": 0.0})

        component_sum = sum(
            v["weighted"] for k, v in breakdown.items()
            if isinstance(v, dict) and "weighted" in v
        )
        assert breakdown["total_before_norm"] == pytest.approx(component_sum)

    def test_factory_builds_all_enabled(self):
        config = _default_config()
        engine = build_reward_engine(config)
        assert "profit" in engine.components
        assert "holding" in engine.components
        assert "drawdown" in engine.components
        assert "transaction" in engine.components
        assert "scaling" in engine.components
        assert "liquidation" in engine.components
        assert "margin" in engine.components
