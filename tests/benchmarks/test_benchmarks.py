"""Tests for benchmark strategies — smoke and correctness."""

import numpy as np
import pandas as pd
import pytest

from src.benchmarks.buy_and_hold import BuyAndHoldBenchmark
from src.benchmarks.mean_reversion import MeanReversionBenchmark
from src.benchmarks.momentum import MomentumBenchmark
from src.benchmarks.random_policy import RandomPolicyBenchmark
from src.environment.registry import make_env
from src.workflows.benchmark_workflow import run_benchmark_workflow


def _make_synthetic_df(n: int = 100, base_price: float = 1.10) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    close = base_price + np.cumsum(rng.randn(n) * 0.0005)
    return pd.DataFrame({
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


def _make_config() -> dict:
    return {
        "environment": {
            "observation": {"window_length": 5, "include_portfolio_state": True, "include_legal_action_mask": True},
            "account": {"initial_capital": 100000.0, "currency": "USD"},
            "leverage": {"max_leverage": 30, "initial_margin_ratio": 0.0333, "maintenance_margin_ratio": 0.5},
            "actions": {
                "base_lot_size": 0.1, "reduce_fraction": 0.5,
                "max_total_position_size": 2.0, "max_open_lots_per_direction": 2.0,
                "max_pyramid_levels": 2, "max_martingale_steps": 2,
                "adverse_threshold": 0.001, "profit_threshold": 0.001,
                "pyramid_increment_lots": 0.1, "martingale_multiplier": 2.0,
                "simplified_mode": False,
            },
            "invalid_action": {"policy": "convert_to_hold_with_penalty", "penalty": 0.0},
            "liquidation": {"threshold": 0.25, "force_close_policy": "full_close"},
            "execution": {"fill_price_rule": "next_bar_open", "mark_price_rule": "next_bar_close"},
            "session": {"timezone": "UTC", "enabled": True},
            "rollover": {"enabled": False},
            "slippage": {"enabled": True, "mode": "deterministic", "base_slippage_pips": 0.5, "volatility_multiplier": 1.0, "session_multiplier": 1.0},
            "transaction_costs": {"enabled": True, "spread_mode": "fixed", "fixed_spread_pips": 1.0, "commission_per_lot": 3.5, "cost_multiplier": 1.0},
            "episode": {"start_policy": "beginning", "end_policy": "end_of_data", "max_steps": None},
        },
        "reward": {
            "components": {"profit": {"enabled": True}},
            "weights": {"profit": 1.0},
            "normalizer": {"mode": "clip_only", "clip_min": -10.0, "clip_max": 10.0},
        },
    }


def _run_benchmark(benchmark, df, pair, config, max_steps=80):
    env = make_env(df, pair, config)
    obs, info = env.reset(seed=42)
    benchmark.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = benchmark.act(obs, info, steps)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    return steps


class TestBuyAndHold:
    def test_runs_to_completion(self):
        df = _make_synthetic_df()
        cfg = _make_config()
        bench = BuyAndHoldBenchmark({"benchmark": {"name": "buy_and_hold"}})
        steps = _run_benchmark(bench, df, "EURUSD", cfg)
        assert steps > 0

    def test_opens_long(self):
        df = _make_synthetic_df()
        cfg = _make_config()
        bench = BuyAndHoldBenchmark({"benchmark": {"name": "buy_and_hold"}})
        env = make_env(df, "EURUSD", cfg)
        obs, info = env.reset(seed=42)
        bench.reset()
        action = bench.act(obs, info, 0)
        assert action == 1  # OPEN_LONG


class TestMeanReversion:
    def test_runs_to_completion(self):
        df = _make_synthetic_df(n=150)
        cfg = _make_config()
        bench = MeanReversionBenchmark({"benchmark": {"name": "mean_reversion", "lookback_window": 10, "entry_z_threshold": 1.5, "exit_z_threshold": 0.5}})
        steps = _run_benchmark(bench, df, "EURUSD", cfg, max_steps=120)
        assert steps > 0


class TestMomentum:
    def test_runs_to_completion(self):
        df = _make_synthetic_df(n=150)
        cfg = _make_config()
        bench = MomentumBenchmark({"benchmark": {"name": "momentum", "lookback_window": 10, "entry_threshold": 0.0}})
        steps = _run_benchmark(bench, df, "EURUSD", cfg, max_steps=120)
        assert steps > 0


class TestRandomPolicy:
    def test_runs_to_completion(self):
        df = _make_synthetic_df()
        cfg = _make_config()
        bench = RandomPolicyBenchmark({"benchmark": {"name": "random_policy", "random_seed": 42}})
        steps = _run_benchmark(bench, df, "EURUSD", cfg)
        assert steps > 0

    def test_is_reproducible(self):
        df = _make_synthetic_df()
        cfg = _make_config()
        results = []
        for _ in range(2):
            bench = RandomPolicyBenchmark({"benchmark": {"name": "random_policy", "random_seed": 0}})
            env = make_env(df, "EURUSD", cfg)
            obs, info = env.reset(seed=42)
            bench.reset()
            actions = []
            for s in range(10):
                a = bench.act(obs, info, s)
                actions.append(a)
                obs, _, term, trunc, info = env.step(a)
                if term or trunc:
                    break
            results.append(actions)
        assert results[0] == results[1]

    def test_only_legal_actions(self):
        df = _make_synthetic_df()
        cfg = _make_config()
        bench = RandomPolicyBenchmark({"benchmark": {"name": "random_policy", "random_seed": 7}})
        env = make_env(df, "EURUSD", cfg)
        obs, info = env.reset(seed=42)
        bench.reset()
        for s in range(30):
            mask = obs["mask"]
            action = bench.act(obs, info, s)
            assert mask[action] > 0, f"Illegal action {action} at step {s}"
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break


class TestBenchmarkWorkflowArtifacts:
    def test_workflow_writes_canonical_outputs(self, tmp_path):
        df = _make_synthetic_df(n=120)
        cfg = _make_config()

        results = run_benchmark_workflow(
            config=cfg,
            benchmark_name="buy_and_hold",
            benchmark_config={},
            pairs=["EURUSD"],
            data={"EURUSD": df},
            outputs_root=str(tmp_path),
            split_name="test",
        )

        assert "EURUSD" in results

        run_dir = tmp_path / "results" / "benchmarks" / "buy_and_hold" / "EURUSD"
        assert (run_dir / "resolved_config.yaml").exists()
        assert (run_dir / "metrics" / "test" / "cumulative_return.csv").exists()
        assert (run_dir / "metrics" / "test" / "sharpe_ratio.csv").exists()
        assert (run_dir / "metrics" / "test" / "max_drawdown.csv").exists()
        assert (run_dir / "metrics" / "test" / "turnover.csv").exists()
        assert (run_dir / "metrics" / "test" / "actions_sequence.csv").exists()
        assert (run_dir / "metrics" / "test" / "trade_log.csv").exists()
        assert (run_dir / "tables" / "test" / "performance_summary.csv").exists()
        assert (run_dir / "tables" / "test" / "risk_metrics.csv").exists()
        assert any((run_dir / "figures" / "test").glob("equity_curve.*"))
        assert any((run_dir / "figures" / "test").glob("drawdown_curve.*"))
        assert (run_dir / "logs" / "evaluation.log").exists()
