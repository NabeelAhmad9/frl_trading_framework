"""Artifact schema validation — verify canonical output structure after a training run."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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


def _base_config() -> dict:
    return {
        "training": {
            "random_seed": 42,
            "total_timesteps": 60,
            "max_episode_steps": None,
            "evaluation_interval": 100,
            "checkpoint_interval": 30,
            "warmup_steps": 5,
            "device": {"preference": "cpu"},
            "resume": {"enabled": False},
            "curriculum": {"enabled": False},
            "trainer": {"log_interval": 30},
        },
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
            "replay": {"buffer_size": 200, "batch_size": 8},
            "target": {"update_interval": 30},
            "exploration": {
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay_steps": 50,
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
            "components": {
                "profit": {"enabled": True},
                "holding": {"enabled": False},
                "volatility": {"enabled": False},
                "drawdown": {"enabled": True},
                "transaction": {"enabled": True},
                "overtrading": {"enabled": False},
                "scaling": {"enabled": False},
                "margin": {"enabled": False},
                "liquidation": {"enabled": False},
            },
            "weights": {"profit": 1.0, "drawdown": 0.2, "transaction": 1.0},
            "normalizer": {"mode": "clip_only", "clip_min": -10.0, "clip_max": 10.0},
        },
        "logging": {
            "level": "WARNING",
            "console": {"enabled": False},
            "file": {"enabled": False},
            "log_rewards_by_component": False,
            "evaluation_verbosity": "quiet",
        },
    }


@pytest.fixture(scope="module")
def trained_run_dir():
    """Train a tiny DQN agent and return the run directory."""
    from src.training.trainer import Trainer

    tmpdir = tempfile.mkdtemp()
    config = _base_config()
    df = _make_synthetic_df(n=100)
    run_dir = Path(tmpdir) / "run"

    trainer = Trainer(df=df, pair="EURUSD", config=config, run_dir=run_dir)
    trainer.train()
    return run_dir


class TestArtifactSchema:
    def test_checkpoints_dir_exists(self, trained_run_dir):
        assert (trained_run_dir / "checkpoints").is_dir()

    def test_checkpoint_latest_exists(self, trained_run_dir):
        assert (trained_run_dir / "checkpoints" / "checkpoint_latest.pt").is_file()

    def test_training_state_json(self, trained_run_dir):
        state_json = trained_run_dir / "checkpoints" / "training_state.json"
        assert state_json.is_file()
        import json
        with open(state_json) as f:
            state = json.load(f)
        assert "episode" in state

    def test_models_dir_has_final(self, trained_run_dir):
        model_path = trained_run_dir / "models" / "model_final.pt"
        assert model_path.is_file()

    def test_metrics_dir_exists(self, trained_run_dir):
        assert (trained_run_dir / "metrics").is_dir()

    def test_reward_curve_csv(self, trained_run_dir):
        rc = trained_run_dir / "metrics" / "train" / "reward_curve.csv"
        assert rc.is_file()
        df = pd.read_csv(rc)
        assert "reward_total" in df.columns

    def test_loss_curve_csv(self, trained_run_dir):
        lc = trained_run_dir / "metrics" / "train" / "loss_curve.csv"
        assert lc.is_file()
        df = pd.read_csv(lc)
        assert "td_loss" in df.columns

    def test_train_metric_snapshots_exist(self, trained_run_dir):
        metrics_dir = trained_run_dir / "metrics" / "train"
        assert (metrics_dir / "cumulative_return.csv").is_file()
        assert (metrics_dir / "sharpe_ratio.csv").is_file()
        assert (metrics_dir / "max_drawdown.csv").is_file()
        assert (metrics_dir / "turnover.csv").is_file()

    def test_train_tables_exist(self, trained_run_dir):
        tables_dir = trained_run_dir / "tables" / "train"
        assert (tables_dir / "performance_summary.csv").is_file()
        assert (tables_dir / "risk_metrics.csv").is_file()

    def test_train_figures_exist(self, trained_run_dir):
        figures_dir = trained_run_dir / "figures" / "train"
        assert any(figures_dir.glob("equity_curve.*"))
        assert any(figures_dir.glob("drawdown_curve.*"))

    def test_logs_dir_exists(self, trained_run_dir):
        assert (trained_run_dir / "logs").is_dir()

    def test_training_log_exists(self, trained_run_dir):
        assert (trained_run_dir / "logs" / "training.log").is_file()
