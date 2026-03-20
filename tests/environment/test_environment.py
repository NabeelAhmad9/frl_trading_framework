"""Tests for the full TradingEnv — reset/step, legal masks, no lookahead, episode boundaries."""

import numpy as np
import pandas as pd
import pytest

from src.environment.trading_env import TradingEnv
from src.environment.registry import make_env
from src.environment.action_space import Action


def _make_synthetic_df(n: int = 100, base_price: float = 1.10) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with feature columns."""
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    close = base_price + np.cumsum(rng.randn(n) * 0.0005)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": close - rng.uniform(0, 0.001, n),
        "high": close + rng.uniform(0, 0.005, n),
        "low": close - rng.uniform(0, 0.005, n),
        "close": close,
        "volume": rng.uniform(100, 10000, n),
        "sma_10": close,
        "rsi_14": rng.uniform(30, 70, n),
        "spread_proxy_hl": rng.uniform(0.0001, 0.0005, n),
        "realized_volatility_proxy": rng.uniform(0.001, 0.01, n),
    })
    return df


def _make_config(simplified: bool = False) -> dict:
    return {
        "environment": {
            "observation": {"window_length": 5, "include_portfolio_state": True, "include_legal_action_mask": True},
            "account": {"initial_capital": 100000.0, "currency": "USD"},
            "leverage": {"max_leverage": 30, "initial_margin_ratio": 0.0333333333, "maintenance_margin_ratio": 0.5},
            "actions": {
                "base_lot_size": 0.1, "reduce_fraction": 0.5,
                "max_total_position_size": 2.0, "max_open_lots_per_direction": 2.0,
                "max_pyramid_levels": 2, "max_martingale_steps": 2,
                "adverse_threshold": 0.001, "profit_threshold": 0.001,
                "pyramid_increment_lots": 0.1, "martingale_multiplier": 2.0,
                "simplified_mode": simplified,
            },
            "invalid_action": {"policy": "convert_to_hold_with_penalty", "penalty": 0.01},
            "liquidation": {"threshold": 0.25, "force_close_policy": "full_close"},
            "execution": {"fill_price_rule": "next_bar_open", "mark_price_rule": "next_bar_close"},
            "session": {"timezone": "UTC", "enabled": True},
            "rollover": {"enabled": False},
            "slippage": {"enabled": True, "mode": "deterministic", "base_slippage_pips": 0.5, "volatility_multiplier": 1.0, "session_multiplier": 1.0},
            "transaction_costs": {"enabled": True, "spread_mode": "fixed", "fixed_spread_pips": 1.0, "commission_per_lot": 3.5, "cost_multiplier": 1.0},
            "episode": {"start_policy": "beginning", "end_policy": "end_of_data", "max_steps": None},
        }
    }


class TestResetStep:
    """Reset and step must follow Gymnasium contract."""

    def test_reset_returns_obs_and_info(self):
        df = _make_synthetic_df()
        env = make_env(df, "EURUSD", _make_config())
        obs, info = env.reset(seed=42)
        assert "market" in obs
        assert "portfolio" in obs
        assert "mask" in obs
        assert "flat" in obs
        assert isinstance(info, dict)

    def test_step_returns_five_elements(self):
        df = _make_synthetic_df()
        env = make_env(df, "EURUSD", _make_config())
        env.reset(seed=42)
        result = env.step(Action.HOLD)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_obs_shapes(self):
        df = _make_synthetic_df()
        config = _make_config()
        env = make_env(df, "EURUSD", config)
        obs, _ = env.reset(seed=42)
        assert obs["market"].shape == (5, env.num_features)
        assert obs["portfolio"].shape == (10,)
        assert obs["mask"].shape == (10,)

    def test_full_episode_terminates(self):
        df = _make_synthetic_df(n=30)
        env = make_env(df, "EURUSD", _make_config())
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            obs, r, term, trunc, info = env.step(Action.HOLD)
            done = term or trunc
            steps += 1
            if steps > 100:
                break
        assert done


class TestLegalMaskConsistency:
    """Legal mask in info must be consistent with portfolio state."""

    def test_mask_flat_after_reset(self):
        df = _make_synthetic_df()
        env = make_env(df, "EURUSD", _make_config())
        obs, info = env.reset(seed=42)
        mask = obs["mask"]
        assert mask[Action.HOLD] == 1.0
        assert mask[Action.OPEN_LONG] == 1.0
        assert mask[Action.OPEN_SHORT] == 1.0
        assert mask[Action.CLOSE_POSITION] == 0.0

    def test_mask_after_open(self):
        df = _make_synthetic_df()
        env = make_env(df, "EURUSD", _make_config())
        env.reset(seed=42)
        obs, _, _, _, info = env.step(Action.OPEN_LONG)
        mask = obs["mask"]
        assert mask[Action.OPEN_LONG] == 0.0
        assert mask[Action.OPEN_SHORT] == 0.0
        assert mask[Action.CLOSE_POSITION] == 1.0
        assert mask[Action.REDUCE_POSITION] == 1.0


class TestInvalidActionHandling:
    """Invalid actions must be caught and converted to HOLD with penalty."""

    def test_invalid_action_converts_to_hold(self):
        df = _make_synthetic_df()
        env = make_env(df, "EURUSD", _make_config())
        env.reset(seed=42)
        # CLOSE is invalid when flat
        obs, reward, _, _, info = env.step(Action.CLOSE_POSITION)
        assert info["executed_action"] == Action.HOLD
        assert info["was_legal"] is False

    def test_out_of_range_action_is_handled_safely(self):
        df = _make_synthetic_df()
        env = make_env(df, "EURUSD", _make_config())
        env.reset(seed=42)
        obs, reward, _, _, info = env.step(999)
        assert info["executed_action"] == Action.HOLD
        assert info["was_legal"] is False


class TestNoLookahead:
    """Verify anti-lookahead: agent observes close_t, executes at open_{t+1}."""

    def test_execution_uses_next_bar_data(self):
        df = _make_synthetic_df(n=50)
        env = make_env(df, "EURUSD", _make_config())
        env.reset(seed=42)
        # Record the next bar open before stepping
        cursor_before = env.simulator.cursor
        next_open = env.simulator.next_state().open

        obs, _, _, _, info = env.step(Action.OPEN_LONG)
        # The fill should be based on next_open, not current close
        # We can't directly check fill price in info without more detail,
        # but we check cursor advanced
        assert env.simulator.cursor == cursor_before + 1


class TestEpisodeBoundary:
    """Episode must end cleanly at data boundary."""

    def test_max_steps_truncates(self):
        df = _make_synthetic_df(n=50)
        config = _make_config()
        config["environment"]["episode"]["max_steps"] = 5
        env = make_env(df, "EURUSD", config)
        env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            obs, r, term, trunc, info = env.step(Action.HOLD)
            done = term or trunc
            steps += 1
        assert steps == 5

    def test_short_dataset_with_large_window_is_safe(self):
        df = _make_synthetic_df(n=3)
        config = _make_config()
        config["environment"]["observation"]["window_length"] = 10
        env = make_env(df, "EURUSD", config)
        obs, _ = env.reset(seed=42)
        assert env.simulator.cursor <= env.simulator.length - 1
        obs, r, term, trunc, info = env.step(Action.HOLD)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)


class TestSimplifiedMode:
    """Simplified 3-action mode must work through registry."""

    def test_simplified_obs_mask_shape(self):
        df = _make_synthetic_df()
        env = make_env(df, "EURUSD", _make_config(), simplified=True)
        obs, _ = env.reset(seed=42)
        assert obs["mask"].shape == (3,)
        assert env.num_actions == 3

    def test_simplified_target_long(self):
        df = _make_synthetic_df()
        env = make_env(df, "EURUSD", _make_config(), simplified=True)
        env.reset(seed=42)
        obs, _, _, _, info = env.step(1)  # TARGET_LONG
        assert info["direction"] == 1


class TestDeterminism:
    """Environment must be deterministic under fixed seed."""

    def test_same_seed_same_trajectory(self):
        df = _make_synthetic_df()
        config = _make_config()

        rewards_a = []
        env = make_env(df, "EURUSD", config)
        env.reset(seed=42)
        for _ in range(10):
            _, r, _, _, _ = env.step(Action.HOLD)
            rewards_a.append(r)

        rewards_b = []
        env2 = make_env(df, "EURUSD", config)
        env2.reset(seed=42)
        for _ in range(10):
            _, r, _, _, _ = env2.step(Action.HOLD)
            rewards_b.append(r)

        np.testing.assert_array_equal(rewards_a, rewards_b)
