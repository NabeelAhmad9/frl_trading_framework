"""Tests for the training workflow — config, smoke training, artifacts, resume."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training.trainer import Trainer
from src.training.checkpoint_manager import CheckpointManager
from src.training.resume_manager import ResumeManager
from src.training.epsilon_scheduler import EpsilonScheduler
from src.training.reward_logger import RewardLogger
from src.training.seed_manager import init_seeds
from src.training.curriculum_scheduler import CurriculumScheduler
from src.utils.config_loader import resolve_config


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


def _make_config(total_timesteps: int = 200, checkpoint_interval: int = 100) -> dict:
    return {
        "training": {
            "random_seed": 42,
            "total_timesteps": total_timesteps,
            "max_episode_steps": None,
            "evaluation_interval": 100,
            "checkpoint_interval": checkpoint_interval,
            "warmup_steps": 10,
            "device": {"preference": "cpu"},
            "resume": {"enabled": True},
            "curriculum": {"enabled": False},
            "trainer": {"log_interval": 50},
        },
        "agent": {
            "name": "dqn",
            "algorithm": {"double_dqn": False},
            "model": {
                "encoder_type": "mlp",
                "hidden_dims": [32, 16],
                "dueling": False,
                "activation": "relu",
                "dropout": 0.0,
            },
            "optimizer": {"learning_rate": 1e-3},
            "discount": {"gamma": 0.99},
            "replay": {"buffer_size": 500, "batch_size": 16},
            "target": {"update_interval": 50},
            "exploration": {
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay_steps": 150,
            },
            "training": {
                "gradient_clip_norm": 10.0,
                "learn_start_steps": 10,
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
            "log_rewards_by_component": True,
            "reward_component_log_interval": 50,
            "evaluation_verbosity": "quiet",
            "log_system_info": False,
        },
    }


# ============================================================
# Seed manager
# ============================================================
class TestSeedManager:
    def test_init_seeds(self):
        cfg = {"training": {"random_seed": 99}}
        seed = init_seeds(cfg)
        assert seed == 99


# ============================================================
# Epsilon scheduler
# ============================================================
class TestEpsilonScheduler:
    def test_linear_decay(self):
        sched = EpsilonScheduler(start=1.0, end=0.0, decay_steps=10)
        for _ in range(10):
            sched.step()
        assert abs(sched.epsilon - 0.0) < 1e-6

    def test_does_not_go_below_end(self):
        sched = EpsilonScheduler(start=1.0, end=0.1, decay_steps=5)
        for _ in range(20):
            sched.step()
        assert sched.epsilon >= 0.1 - 1e-9

    def test_set_step(self):
        sched = EpsilonScheduler(start=1.0, end=0.0, decay_steps=100)
        sched.set_step(50)
        assert abs(sched.epsilon - 0.5) < 1e-6


# ============================================================
# Curriculum scheduler
# ============================================================
class TestCurriculumScheduler:
    def test_disabled_by_default(self):
        cs = CurriculumScheduler({"training": {"curriculum": {"enabled": False}}})
        assert not cs.should_advance(100)
        assert cs.get_current_difficulty() == {}


# ============================================================
# Reward logger
# ============================================================
class TestRewardLogger:
    def test_creates_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rl = RewardLogger(Path(tmpdir), log_components=True)
            rl.log_episode_reward(100, 1, 5.0, {"profit": 3.0, "drawdown": -1.0})
            rl.log_episode_metrics(100, 1, {
                "cumulative_return": 0.1,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.05,
                "turnover": 0.3,
            })
            rl.log_loss(100, 0.05)
            rl.log_action(100, {"raw_action": 1, "action_name": "OPEN_LONG", "was_legal": True})
            rl.close()
            assert (Path(tmpdir) / "reward_curve.csv").exists()
            assert (Path(tmpdir) / "loss_curve.csv").exists()
            assert (Path(tmpdir) / "actions_sequence.csv").exists()
            assert (Path(tmpdir) / "cumulative_return.csv").exists()
            assert (Path(tmpdir) / "sharpe_ratio.csv").exists()
            assert (Path(tmpdir) / "max_drawdown.csv").exists()
            assert (Path(tmpdir) / "turnover.csv").exists()

    def test_reward_schema_expands_when_new_components_appear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rl = RewardLogger(Path(tmpdir), log_components=True)
            rl.log_episode_reward(1, 1, 1.0, None)
            rl.log_episode_reward(2, 2, 2.0, {"profit": 1.5, "drawdown": -0.2})
            rl.close()

            df = pd.read_csv(Path(tmpdir) / "reward_curve.csv")
            assert "reward_profit" in df.columns
            assert "reward_drawdown" in df.columns
            row = df[df["episode"] == 2].iloc[0]
            assert row["reward_profit"] == pytest.approx(1.5)
            assert row["reward_drawdown"] == pytest.approx(-0.2)

    def test_append_mode_preserves_existing_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir)

            rl1 = RewardLogger(metrics_dir, log_components=True, append_mode=False)
            rl1.log_episode_reward(10, 1, 1.0, {"profit": 1.0})
            rl1.log_action(10, {"raw_action": 1, "action_name": "OPEN_LONG", "was_legal": True})
            rl1.log_trade_event(10, 1, {
                "step": 1,
                "pnl": 12.5,
                "direction_before": 0,
                "direction": 1,
                "forced_liquidation": False,
                "pyramid_steps": 0,
                "martingale_steps": 0,
                "notional": 10000.0,
            })
            rl1.close()

            rl2 = RewardLogger(metrics_dir, log_components=True, append_mode=True)
            rl2.log_episode_reward(20, 2, 2.0, {"profit": 2.0, "drawdown": -0.1})
            rl2.log_action(20, {"raw_action": 2, "action_name": "OPEN_SHORT", "was_legal": True})
            rl2.log_trade_event(20, 2, {
                "step": 2,
                "pnl": -3.5,
                "direction_before": 1,
                "direction": 0,
                "forced_liquidation": False,
                "pyramid_steps": 1,
                "martingale_steps": 0,
                "notional": 8000.0,
            })
            rl2.close()

            reward_df = pd.read_csv(metrics_dir / "reward_curve.csv")
            action_df = pd.read_csv(metrics_dir / "actions_sequence.csv")
            trade_df = pd.read_csv(metrics_dir / "trade_log.csv")

            assert len(reward_df) == 2
            assert len(action_df) == 2
            assert len(trade_df) == 2
            assert reward_df["episode"].tolist() == [1, 2]


# ============================================================
# Checkpoint + resume
# ============================================================
class TestCheckpointResume:
    def test_save_and_detect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.agents.dqn.dqn_agent import DQNAgent
            agent = DQNAgent(10, 3, _make_config(), device="cpu")
            mgr = CheckpointManager(Path(tmpdir))
            mgr.save(agent, global_step=50, episode=2, epsilon=0.5, config=_make_config())
            assert mgr.has_checkpoint()
            assert (Path(tmpdir) / "training_state.json").exists()

            with open(Path(tmpdir) / "training_state.json") as f:
                state = json.load(f)
            assert state["global_step"] == 50

    def test_resume_restores_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.agents.dqn.dqn_agent import DQNAgent
            cfg = _make_config()
            agent = DQNAgent(10, 3, cfg, device="cpu")
            mgr = CheckpointManager(Path(tmpdir))
            mgr.save(agent, global_step=100, episode=5, epsilon=0.3, config=cfg)

            agent2 = DQNAgent(10, 3, cfg, device="cpu")
            rm = ResumeManager(mgr)
            assert rm.can_resume()
            step, ep, state = rm.restore(agent2)
            assert step == 100
            assert ep == 5

    def test_restore_includes_runtime_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.agents.dqn.dqn_agent import DQNAgent

            cfg = _make_config()
            agent = DQNAgent(10, 3, cfg, device="cpu")
            mgr = CheckpointManager(Path(tmpdir))
            runtime_state = {
                "epsilon_scheduler": {"step": 12, "epsilon": 0.88},
                "replay": {"capacity": 500, "pos": 0, "storage": [], "rng_state": np.random.RandomState(42).get_state()},
            }
            mgr.save(agent, global_step=40, episode=3, epsilon=0.7, config=cfg, runtime_state=runtime_state)

            agent2 = DQNAgent(10, 3, cfg, device="cpu")
            rm = ResumeManager(mgr)
            _, _, state = rm.restore(agent2)
            assert "runtime_state" in state
            assert state["runtime_state"]["epsilon_scheduler"]["step"] == 12


# ============================================================
# Smoke training
# ============================================================
class TestSmokeTraining:
    def test_short_training_completes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = _make_synthetic_df(n=200)
            cfg = _make_config(total_timesteps=150, checkpoint_interval=75)
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir()

            trainer = Trainer(df=df, pair="EURUSD", config=cfg, run_dir=run_dir)
            summary = trainer.train()

            assert summary["total_steps"] >= 150
            assert summary["total_episodes"] >= 1
            assert "mean_loss" in summary

            # Artifacts created
            assert (run_dir / "resolved_config.yaml").exists()
            ckpt_dir = run_dir / "checkpoints"
            assert ckpt_dir.exists()
            assert (ckpt_dir / "checkpoint_latest.pt").exists()
            assert (ckpt_dir / "training_state.json").exists()

            models_dir = run_dir / "models"
            assert (models_dir / "model_final.pt").exists()

            metrics_dir = run_dir / "metrics" / "train"
            assert (metrics_dir / "cumulative_return.csv").exists()
            assert (metrics_dir / "sharpe_ratio.csv").exists()
            assert (metrics_dir / "max_drawdown.csv").exists()
            assert (metrics_dir / "turnover.csv").exists()
            assert (metrics_dir / "trade_log.csv").exists()

            tables_dir = run_dir / "tables" / "train"
            assert (tables_dir / "performance_summary.csv").exists()
            assert (tables_dir / "risk_metrics.csv").exists()

            figures_dir = run_dir / "figures" / "train"
            assert any(figures_dir.glob("equity_curve.*"))
            assert any(figures_dir.glob("drawdown_curve.*"))

            logs_dir = run_dir / "logs"
            assert (logs_dir / "training.log").exists()

    def test_double_dqn_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = _make_synthetic_df(n=200)
            cfg = _make_config(total_timesteps=100, checkpoint_interval=50)
            cfg["agent"]["algorithm"]["double_dqn"] = True
            cfg["agent"]["name"] = "doubledqn"
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir()

            trainer = Trainer(df=df, pair="EURUSD", config=cfg, run_dir=run_dir)
            summary = trainer.train()
            assert summary["total_steps"] >= 100

    def test_resume_continues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = _make_synthetic_df(n=200)
            cfg = _make_config(total_timesteps=80, checkpoint_interval=40)
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir()

            # First run: 80 steps
            trainer = Trainer(df=df, pair="EURUSD", config=cfg, run_dir=run_dir)
            summary1 = trainer.train()
            step1 = summary1["total_steps"]

            # Second run: resume with more steps
            cfg2 = _make_config(total_timesteps=160, checkpoint_interval=40)
            trainer2 = Trainer(df=df, pair="EURUSD", config=cfg2, run_dir=run_dir)
            summary2 = trainer2.train()
            # Should have done more total steps
            assert summary2["total_steps"] >= step1


# ============================================================
# Config resolution regression checks
# ============================================================
class TestConfigResolution:
    def test_resolve_config_normalizes_paths_and_metadata(self):
        root = Path(__file__).resolve().parents[2]
        cfg = resolve_config(root=root, agent_config="configs/agents/dqn.yaml")

        assert Path(cfg["data"]["raw_data_dir"]).is_absolute()
        assert Path(cfg["data"]["processed_data_dir"]).is_absolute()
        assert cfg["meta"]["project_name"] == "frl-trading-framework"
        assert cfg["meta"]["agent_name"] == "dqn"
        assert cfg["paths"]["root"] == str(root.resolve())

    def test_resolve_config_rejects_missing_agent_config(self):
        root = Path(__file__).resolve().parents[2]
        with pytest.raises(FileNotFoundError):
            resolve_config(root=root, agent_config="configs/agents/missing.yaml")

    def test_resolve_config_rejects_unreachable_learning_schedule(self):
        root = Path(__file__).resolve().parents[2]
        with pytest.raises(ValueError, match="total_timesteps"):
            resolve_config(
                root=root,
                agent_config="configs/agents/dqn.yaml",
                cli_overrides={
                    "training": {"total_timesteps": 100, "warmup_steps": 50},
                    "agent": {"training": {"learn_start_steps": 100}},
                },
            )
