"""Smoke test matrix — verify end-to-end workflows complete without errors.

Each test uses tiny synthetic data and minimal timesteps to confirm the
entire pipeline is wired up correctly.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _base_config(agent_name: str = "dqn", double: bool = False) -> dict:
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
            "name": agent_name,
            "algorithm": {"double_dqn": double},
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
        "evaluation": {
            "deterministic_policy": True,
            "periods_per_year": 6048,
            "risk_free_rate": 0.0,
            "figure_format": "png",
        },
        "logging": {
            "level": "WARNING",
            "console": {"enabled": False},
            "file": {"enabled": False},
            "log_rewards_by_component": False,
            "evaluation_verbosity": "quiet",
        },
    }


# ---------------------------------------------------------------------------
# 1. Short environment episode
# ---------------------------------------------------------------------------

class TestSmokeEnvironment:
    def test_env_episode_completes(self):
        from src.environment.registry import make_env
        from src.reward.reward_factory import build_reward_engine

        config = _base_config()
        df = _make_synthetic_df(n=80)
        reward_engine = build_reward_engine(config)
        env = make_env(df, "EURUSD", config, reward_engine=reward_engine)
        obs, info = env.reset(seed=42)
        assert "flat" in obs

        done = False
        steps = 0
        while not done and steps < 50:
            obs, reward, terminated, truncated, info = env.step(0)  # HOLD
            done = terminated or truncated
            steps += 1
        assert steps > 0


# ---------------------------------------------------------------------------
# 2. Short DQN training run
# ---------------------------------------------------------------------------

class TestSmokeDQN:
    def test_dqn_trains(self):
        from src.training.trainer import Trainer
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _base_config("dqn", double=False)
            df = _make_synthetic_df(n=100)
            trainer = Trainer(df=df, pair="EURUSD", config=config, run_dir=Path(tmpdir))
            summary = trainer.train()
            assert summary is not None
            assert (Path(tmpdir) / "checkpoints").exists()


# ---------------------------------------------------------------------------
# 3. Short Double DQN training run
# ---------------------------------------------------------------------------

class TestSmokeDoubleDQN:
    def test_ddqn_trains(self):
        from src.training.trainer import Trainer
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _base_config("doubledqn", double=True)
            df = _make_synthetic_df(n=100)
            trainer = Trainer(df=df, pair="EURUSD", config=config, run_dir=Path(tmpdir))
            summary = trainer.train()
            assert summary is not None


# ---------------------------------------------------------------------------
# 4. Evaluation workflow
# ---------------------------------------------------------------------------

class TestSmokeEvaluation:
    def test_evaluate_workflow(self):
        from src.training.trainer import Trainer
        from src.workflows.evaluate_workflow import run_evaluate_workflow
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _base_config("dqn", double=False)
            df = _make_synthetic_df(n=100)
            run_dir = Path(tmpdir) / "run"
            trainer = Trainer(df=df, pair="EURUSD", config=config, run_dir=run_dir)
            trainer.train()

            cp_path = str(run_dir / "checkpoints" / "checkpoint_latest.pt")
            eval_df = _make_synthetic_df(n=80)
            metrics = run_evaluate_workflow(
                config=config, pair="EURUSD", eval_df=eval_df,
                run_dir=run_dir, checkpoint_path=cp_path,
            )
            assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# 5. Benchmark workflow
# ---------------------------------------------------------------------------

class TestSmokeBenchmarks:
    def test_benchmark_buy_and_hold(self):
        from src.workflows.benchmark_workflow import run_benchmark_workflow
        config = _base_config()
        df = _make_synthetic_df(n=80)
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_benchmark_workflow(
                config=config,
                benchmark_name="buy_and_hold",
                benchmark_config={},
                pairs=["EURUSD"],
                data={"EURUSD": df},
                outputs_root=tmpdir,
            )
            assert "EURUSD" in results


# ---------------------------------------------------------------------------
# 6. Train workflow (orchestration layer)
# ---------------------------------------------------------------------------

class TestSmokeTrainWorkflow:
    def test_train_workflow_completes(self):
        from src.workflows.train_workflow import run_train_workflow
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _base_config("dqn", double=False)
            df = _make_synthetic_df(n=100)
            summary = run_train_workflow(
                config=config, pair="EURUSD", train_df=df, outputs_root=tmpdir,
            )
            assert summary is not None


# ---------------------------------------------------------------------------
# 7. Experiment registry + variant builder
# ---------------------------------------------------------------------------

class TestSmokeExperiments:
    def test_experiment_registry_discovers(self):
        from src.experiments.registry import discover_experiments
        experiments_root = PROJECT_ROOT / "configs" / "experiments"
        families = discover_experiments(experiments_root)
        assert len(families) == 5

    def test_variant_resolution(self):
        from src.experiments.registry import get_experiment
        from src.experiments.variant_builder import resolve_variant
        experiments_root = PROJECT_ROOT / "configs" / "experiments"
        exp = get_experiment(experiments_root, "01_reward_ablation")
        config = resolve_variant(PROJECT_ROOT, exp["metadata"], exp["variants"][0]["path"])
        assert "experiment" in config
        assert config["experiment"]["base_agent"] == "doubledqn"
