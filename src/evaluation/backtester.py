"""Backtester — run deterministic evaluation episodes using trained agents."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.environment.registry import make_env
from src.reward.reward_factory import build_reward_engine
from src.agents.dqn.dqn_agent import DQNAgent
from src.agents.doubledqn.doubledqn_agent import DoubleDQNAgent
from src.evaluation.trade_log import build_trade_event
from src.models.model_factory import build_model
from src.utils.logger import get_logger
from src.utils.progress import (
    get_progress_colour,
    progress_bar,
    progress_logging_redirect,
    set_bar_metrics,
)

logger = get_logger(__name__)


def _build_agent(input_dim: int, num_actions: int, config: dict, device: str):
    agent_cfg = config.get("agent", {})
    is_double = agent_cfg.get("algorithm", {}).get("double_dqn", False)
    if is_double:
        return DoubleDQNAgent(input_dim, num_actions, config, device=device)
    return DQNAgent(input_dim, num_actions, config, device=device)


class BacktestResult:
    """Container for evaluation results."""

    def __init__(self):
        self.equity_curve: List[float] = []
        self.actions: List[Dict[str, Any]] = []
        self.trade_log: List[Dict[str, Any]] = []
        self.total_reward: float = 0.0
        self.steps: int = 0


def run_backtest(
    df: pd.DataFrame,
    pair: str,
    config: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    agent=None,
    deterministic: bool = True,
    progress_position_offset: int = 0,
) -> BacktestResult:
    """Run a deterministic evaluation episode.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-ready evaluation data.
    pair : str
    config : dict
        Resolved config.
    checkpoint_path : str, optional
        Path to checkpoint file. If provided and agent is None, loads agent from it.
    agent : optional
        Pre-built agent (overrides checkpoint loading).
    deterministic : bool
        Force greedy policy (epsilon=0).

    Returns
    -------
    BacktestResult
    """
    reward_engine = build_reward_engine(config)
    env = make_env(df, pair, config, reward_engine=reward_engine)

    if agent is None:
        obs, _ = env.reset(seed=42)
        flat_dim = obs["flat"].shape[0]
        num_actions = env.action_space.n
        agent = _build_agent(flat_dim, num_actions, config, device="cpu")
        if checkpoint_path:
            agent.load_checkpoint(checkpoint_path)

    agent.eval_mode()
    if deterministic:
        agent.epsilon = 0.0

    obs, info = env.reset(seed=42)
    result = BacktestResult()
    result.equity_curve.append(info.get("equity", info.get("equity_after", 100000.0)))

    start_idx = max(getattr(env, "window_length", 1) - 1, 0)
    sim_length = getattr(env.simulator, "length", 0)
    sim_remaining = max(0, sim_length - 1 - start_idx)
    if getattr(env, "max_steps", None) is not None:
        sim_remaining = min(sim_remaining, int(env.max_steps))
    total_steps_hint = sim_remaining if sim_remaining > 0 else None
    progress_cfg = config.get("logging", {}).get("progress", {})
    metrics_update_interval = max(int(progress_cfg.get("metrics_update_interval", 50)), 1)

    done = False
    with progress_logging_redirect(config):
        with progress_bar(
            config,
            desc=f"Backtest {pair}",
            total=total_steps_hint,
            initial=0,
            position=int(progress_position_offset),
            leave=False,
            unit="step",
            colour=get_progress_colour(config, "evaluation"),
        ) as backtest_pbar:
            while not done:
                flat_obs = obs["flat"]
                mask = obs["mask"]
                action = agent.act(flat_obs, mask, training=False)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                result.total_reward += reward
                result.steps += 1
                backtest_pbar.update(1)
                result.equity_curve.append(info.get("equity", info.get("equity_after", result.equity_curve[-1])))

                result.actions.append({
                    "step": result.steps,
                    "timestamp": info.get("timestamp_decision", ""),
                    "action_id": info.get("raw_action", action),
                    "action_name": info.get("action_name", ""),
                    "was_legal": info.get("was_legal", True),
                    "executed_action_name": info.get("executed_action_name", info.get("action_name", "")),
                    "direction_before": info.get("direction_before", 0),
                    "direction_after": info.get("direction", 0),
                    "total_lots_after": info.get("total_lots", 0),
                })

                trade_event = build_trade_event(info, step=result.steps)
                if trade_event is not None:
                    result.trade_log.append(trade_event)

                if (result.steps % metrics_update_interval) == 0 or done:
                    set_bar_metrics(
                        backtest_pbar,
                        metrics={
                            "reward": f"{result.total_reward:.2f}",
                            "equity": f"{result.equity_curve[-1]:.2f}",
                        },
                        refresh=False,
                    )

    logger.info("Backtest: %d steps, total_reward=%.4f, final_equity=%.2f",
                result.steps, result.total_reward, result.equity_curve[-1])
    return result
