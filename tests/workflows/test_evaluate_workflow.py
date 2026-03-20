"""Tests for evaluation workflow artifact generation."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.workflows.evaluate_workflow import run_evaluate_workflow


def _make_synthetic_df(n: int = 120, base_price: float = 1.10) -> pd.DataFrame:
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
        "agent": {
            "name": "dqn",
            "algorithm": {"double_dqn": False},
            "model": {
                "encoder_type": "mlp",
                "hidden_dims": [16, 8],
                "dueling": False,
                "activation": "relu",
                "dropout": 0.0,
            },
            "optimizer": {"learning_rate": 1e-3},
            "discount": {"gamma": 0.99},
            "replay": {"buffer_size": 128, "batch_size": 8},
            "target": {"update_interval": 20},
            "exploration": {
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay_steps": 100,
            },
            "training": {
                "gradient_clip_norm": 10.0,
                "learn_start_steps": 5,
                "learn_frequency": 2,
            },
        },
        "environment": {
            "observation": {"window_length": 5, "include_portfolio_state": True, "include_legal_action_mask": True},
            "account": {"initial_capital": 100000.0, "currency": "USD"},
            "leverage": {"max_leverage": 30, "initial_margin_ratio": 0.0333333333, "maintenance_margin_ratio": 0.5},
            "actions": {
                "base_lot_size": 0.1,
                "reduce_fraction": 0.5,
                "max_total_position_size": 2.0,
                "max_open_lots_per_direction": 2.0,
                "max_pyramid_levels": 2,
                "max_martingale_steps": 2,
                "adverse_threshold": 0.001,
                "profit_threshold": 0.001,
                "pyramid_increment_lots": 0.1,
                "martingale_multiplier": 2.0,
                "simplified_mode": False,
            },
            "invalid_action": {"policy": "convert_to_hold_with_penalty", "penalty": 0.0},
            "liquidation": {"threshold": 0.25, "force_close_policy": "full_close"},
            "execution": {"fill_price_rule": "next_bar_open", "mark_price_rule": "next_bar_close"},
            "session": {"timezone": "UTC", "enabled": True},
            "rollover": {"enabled": False},
            "slippage": {
                "enabled": True,
                "mode": "deterministic",
                "base_slippage_pips": 0.5,
                "volatility_multiplier": 1.0,
                "session_multiplier": 1.0,
            },
            "transaction_costs": {
                "enabled": True,
                "spread_mode": "fixed",
                "fixed_spread_pips": 1.0,
                "commission_per_lot": 3.5,
                "cost_multiplier": 1.0,
            },
            "episode": {"start_policy": "beginning", "end_policy": "end_of_data", "max_steps": None},
        },
        "reward": {
            "components": {"profit": {"enabled": True}},
            "weights": {"profit": 1.0},
            "normalizer": {"mode": "clip_only", "clip_min": -10.0, "clip_max": 10.0},
        },
        "evaluation": {
            "deterministic_policy": True,
            "periods_per_year": 6048,
            "risk_free_rate": 0.0,
            "figure_format": "pdf",
        },
        "logging": {
            "level": "WARNING",
            "console": {"enabled": False},
            "file": {"enabled": False},
        },
    }


def test_evaluate_workflow_writes_trade_and_action_artifacts(tmp_path: Path):
    cfg = _make_config()
    df = _make_synthetic_df()
    run_dir = tmp_path / "results" / "agents" / "dqn" / "EURUSD"

    metrics = run_evaluate_workflow(
        config=cfg,
        pair="EURUSD",
        eval_df=df,
        run_dir=run_dir,
        checkpoint_path=None,
        split_name="test",
    )

    assert "cumulative_return" in metrics
    metrics_dir = run_dir / "metrics" / "test"
    tables_dir = run_dir / "tables" / "test"

    assert (metrics_dir / "actions_sequence.csv").exists()
    assert (metrics_dir / "trade_log.csv").exists()
    assert (metrics_dir / "equity_curve.csv").exists()
    assert (metrics_dir / "drawdown_curve.csv").exists()
    assert (tables_dir / "performance_summary.csv").exists()
    assert (tables_dir / "risk_metrics.csv").exists()
