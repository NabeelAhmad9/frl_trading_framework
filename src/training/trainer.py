"""Trainer — main training loop with replay, checkpointing, and logging."""

import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.agents.common.replay_buffer import ReplayBuffer
from src.agents.dqn.dqn_agent import DQNAgent
from src.agents.doubledqn.doubledqn_agent import DoubleDQNAgent
from src.environment.action_space import ACTION_NAMES
from src.environment.registry import make_env
from src.evaluation.metrics import compute_metrics
from src.evaluation.trade_log import build_trade_event
from src.evaluation.performance_report import (
    save_drawdown_curve,
    save_equity_curve,
    save_performance_summary,
    save_risk_metrics,
    save_trade_log,
)
from src.evaluation.visualization import plot_drawdown_curve, plot_equity_curve
from src.reward.reward_factory import build_reward_engine
from src.training.checkpoint_manager import CheckpointManager
from src.training.curriculum_scheduler import CurriculumScheduler
from src.training.resume_manager import ResumeManager
from src.training.reward_logger import RewardLogger
from src.training.seed_manager import capture_rng_state, init_seeds, restore_rng_state
from src.utils.artifact_manager import ArtifactManager
from src.utils.config_loader import save_resolved_config
from src.utils.device import get_device
from src.utils.logger import get_logger, write_log_file
from src.utils.progress import (
    get_progress_colour,
    progress_bar,
    progress_logging_redirect,
    set_bar_metrics,
)

logger = get_logger(__name__)


def _build_agent(input_dim: int, num_actions: int, config: dict, device: str):
    """Instantiate the correct agent class based on config."""
    agent_cfg = config.get("agent", {})
    is_double = agent_cfg.get("algorithm", {}).get("double_dqn", False)
    if is_double:
        return DoubleDQNAgent(input_dim, num_actions, config, device=device)
    return DQNAgent(input_dim, num_actions, config, device=device)


class Trainer:
    """Single-run training loop with replay, checkpoints, and reward logging."""

    def __init__(
        self,
        df: pd.DataFrame,
        pair: str,
        config: Dict[str, Any],
        run_dir: Path,
        progress_position_offset: int = 0,
    ):
        self.config = config
        self.pair = pair
        self.run_dir = Path(run_dir)
        self.progress_position_offset = int(progress_position_offset)

        # Resolve training params
        t_cfg = config.get("training", {})
        self.total_timesteps = int(t_cfg.get("total_timesteps", 250000))
        self.warmup_steps = max(int(t_cfg.get("warmup_steps", 5000)), 0)
        self.checkpoint_interval = max(int(t_cfg.get("checkpoint_interval", 25000)), 0)
        self.evaluation_interval = int(t_cfg.get("evaluation_interval", 10000))
        self.log_interval = int(t_cfg.get("trainer", {}).get("log_interval", 1000))
        self.resume_enabled = t_cfg.get("resume", {}).get("enabled", True)

        # Agent training params
        a_cfg = config.get("agent", {})
        buffer_size = int(a_cfg.get("replay", {}).get("buffer_size", 100000))
        self.batch_size = int(a_cfg.get("replay", {}).get("batch_size", 64))
        self.learn_frequency = max(int(a_cfg.get("training", {}).get("learn_frequency", 4)), 1)

        raw_learn_start = max(int(a_cfg.get("training", {}).get("learn_start_steps", self.warmup_steps)), 0)
        if self.warmup_steps > 0 and raw_learn_start > 0:
            requested_learn_start = min(self.warmup_steps, raw_learn_start)
        else:
            requested_learn_start = max(self.warmup_steps, raw_learn_start)

        # Guard against replay becoming dominated by prolonged random-policy warmup.
        replay_fraction_cap = max(int(buffer_size * 0.25), self.batch_size)
        timestep_fraction_cap = max(int(self.total_timesteps * 0.05), self.batch_size)
        learn_start_cap = max(self.batch_size, min(replay_fraction_cap, timestep_fraction_cap))
        self.learn_start = max(self.batch_size, min(requested_learn_start, learn_start_cap))
        self.warmup_steps = self.learn_start

        if self.learn_start != requested_learn_start:
            logger.warning(
                "Adjusted warmup/learn-start to avoid excessive random replay fill: requested=%d, effective=%d, replay_cap=%d",
                requested_learn_start,
                self.learn_start,
                buffer_size,
            )

        # Seeds
        self.seed = init_seeds(config)

        # Device
        device_pref = t_cfg.get("device", {}).get("preference", "auto")
        self.device = get_device(device_pref)

        # Reward engine
        self.reward_engine = build_reward_engine(config)

        # Environment
        self.env = make_env(df, pair, config, reward_engine=self.reward_engine)

        # Determine obs dim from env
        obs, _ = self.env.reset(seed=self.seed)
        flat_obs = obs["flat"]
        input_dim = flat_obs.shape[0]
        num_actions = self.env.action_space.n

        # Agent
        self.agent = _build_agent(input_dim, num_actions, config, device=str(self.device))

        # Replay buffer
        self.replay = ReplayBuffer(capacity=buffer_size, seed=self.seed)

        # Schedulers
        # Exploration / epsilon scheduling is handled inside the agent implementation.
        self.curriculum = CurriculumScheduler(config)

        # Artifact dirs
        am = ArtifactManager(self.run_dir.parent.parent if self.run_dir.name != "outputs" else self.run_dir)
        self.dirs = am.ensure_subdirs(self.run_dir)

        # Checkpoint / resume manager
        self.ckpt_mgr = CheckpointManager(self.dirs.get("checkpoints", self.run_dir / "checkpoints"))
        self.resume_mgr = ResumeManager(self.ckpt_mgr)

        # Reward logger
        log_cfg = config.get("logging", {})
        self.reward_logger = RewardLogger(
            metrics_dir=self.dirs.get("metrics_train", self.run_dir / "metrics" / "train"),
            log_components=log_cfg.get("log_rewards_by_component", True),
            component_interval=log_cfg.get("reward_component_log_interval", 1000),
            append_mode=bool(self.resume_enabled and self.resume_mgr.can_resume()),
            flush_interval=int(log_cfg.get("flush_interval", 128)),
        )
        self.train_log_path = self.dirs.get("logs", self.run_dir / "logs") / "training.log"
        self.figure_format = config.get("evaluation", {}).get("figure_format", "pdf")

    # Episode length and termination are controlled by the environment. The trainer
    # should not attempt to compute or enforce episode limits.

    def _replay_state_dict(self) -> Dict[str, Any]:
        """Create a serializable replay-buffer snapshot for resume."""
        return self.replay.state_dict()

    def _restore_replay_state(self, replay_state: Optional[Dict[str, Any]]) -> None:
        """Restore replay-buffer snapshot if present."""
        if not isinstance(replay_state, dict):
            return

        saved_capacity = int(replay_state.get("capacity", self.replay.capacity))
        if saved_capacity != self.replay.capacity:
            logger.warning(
                "Replay capacity mismatch on resume (saved=%d, current=%d); truncating to current capacity.",
                saved_capacity,
                self.replay.capacity,
            )
        self.replay.load_state_dict(replay_state)

    def _runtime_state_dict(self) -> Dict[str, Any]:
        """Collect resumable runtime state beyond model weights."""
        return {
            "replay": self._replay_state_dict(),
            "curriculum_scheduler": self.curriculum.state_dict(),
            "rng_state": capture_rng_state(),
        }

    def _restore_runtime_state(self, runtime_state: Optional[Dict[str, Any]]) -> None:
        """Restore runtime state previously saved via ``_runtime_state_dict``."""
        if not isinstance(runtime_state, dict):
            return

        self._restore_replay_state(runtime_state.get("replay"))
        self.curriculum.load_state_dict(runtime_state.get("curriculum_scheduler", {}))
        restore_rng_state(runtime_state.get("rng_state", {}))

    def _save_checkpoint(self, global_step: int, episode: int) -> None:
        """Persist model + runtime state at a safe episode boundary."""
        self.ckpt_mgr.save(
            self.agent,
            global_step,
            episode,
            float(getattr(self.agent, "epsilon", 0.0)),
            self.config,
            extra={
                "replay_size": len(self.replay),
                # epsilon scheduling is internal to the agent implementation
                "curriculum_stage": self.curriculum.get_current_stage_name(),
            },
            runtime_state=self._runtime_state_dict(),
        )

    def train(self) -> Dict[str, Any]:
        """Execute the full training loop. Returns summary dict."""
        resume_available = bool(self.resume_enabled and self.resume_mgr.can_resume())

        # Save resolved config before training
        save_resolved_config(self.config, ArtifactManager.resolved_config_path(self.run_dir))
        write_log_file(
            self.train_log_path,
            [
                f"training_start pair={self.pair}",
                f"agent={self.config.get('agent', {}).get('name', '')}",
                f"total_timesteps={self.total_timesteps}",
            ],
            mode="a" if resume_available else "w",
        )

        global_step = 0
        episode = 0

        # Resume if possible
        if self.resume_enabled and self.resume_mgr.can_resume():
            global_step, episode, state = self.resume_mgr.restore(self.agent)
            runtime_state = state.get("runtime_state")
            if runtime_state is not None:
                self._restore_runtime_state(runtime_state)
            logger.info("Resumed training from step %d, episode %d", global_step, episode)

        self.agent.train_mode()
        best_episode_reward = -float("inf")
        total_loss_sum = 0.0
        total_loss_count = 0
        last_episode_metrics: Dict[str, float] = {}
        last_episode_equity = None
        # Accumulate equity values across all episodes for full-training curves
        training_equity = []
        training_trade_log = []
        last_checkpoint_step = global_step
        reward_window = deque(maxlen=2000)
        q_window = deque(maxlen=2000)
        action_counts = np.zeros(max(ACTION_NAMES.keys()) + 1, dtype=np.int64)

        # If we resumed from checkpoint, bootstrap full-training accumulators from
        # already persisted artifacts so final metrics remain continuous.
        if resume_available:
            metrics_train_dir = Path(self.dirs.get("metrics_train", self.run_dir / "metrics" / "train"))
            existing_equity_path = metrics_train_dir / "equity_curve.csv"
            existing_trade_path = metrics_train_dir / "trade_log.csv"

            if existing_equity_path.exists():
                try:
                    eq_df = pd.read_csv(existing_equity_path)
                    if "equity_value" in eq_df.columns:
                        training_equity = [float(v) for v in eq_df["equity_value"].tolist()]
                except Exception:
                    logger.exception("Failed to load existing equity curve for resume continuity")

            if existing_trade_path.exists():
                try:
                    tl_df = pd.read_csv(existing_trade_path)
                    for row in tl_df.to_dict(orient="records"):
                        training_trade_log.append({
                            "step": int(row.get("step", 0)),
                            "pnl": float(row.get("pnl", 0.0)),
                            "direction_before": int(row.get("direction_before", 0)),
                            "direction": int(row.get("direction", 0)),
                            "forced_liquidation": bool(row.get("forced_liquidation", False)),
                            "pyramid_steps": int(row.get("pyramid_steps", 0)),
                            "martingale_steps": int(row.get("martingale_steps", 0)),
                            "notional": float(row.get("notional", 0.0)),
                        })
                except Exception:
                    logger.exception("Failed to load existing trade log for resume continuity")

        logger.info("Starting training: %d timesteps, pair=%s", self.total_timesteps, self.pair)
        t0 = time.time()

        progress_cfg = self.config.get("logging", {}).get("progress", {})
        metrics_update_interval = max(int(progress_cfg.get("metrics_update_interval", 50)), 1)
        train_colour = get_progress_colour(self.config, "training")
        episode_colour = get_progress_colour(self.config, "episode")

        with progress_logging_redirect(self.config):
            with progress_bar(
                self.config,
                desc=f"Training {self.pair}",
                total=self.total_timesteps,
                initial=global_step,
                position=self.progress_position_offset,
                leave=True,
                unit="step",
                colour=train_colour,
            ) as training_pbar:
                while global_step < self.total_timesteps:
                    # Before starting a new episode, avoid beginning one that cannot
                    # possibly complete within the remaining global timesteps. This
                    # prevents creating a truncated final episode when the dataset
                    # is played end-to-end (start_policy=beginning, end_policy=end_of_data)
                    # and no environment max_steps cap is set.
                    remaining_steps = int(self.total_timesteps - global_step)

                    episode_expected: Optional[int] = None
                    try:
                        # If the environment publishes an explicit max_steps, prefer it.
                        if getattr(self.env, "max_steps", None) is not None:
                            episode_expected = int(self.env.max_steps)
                        # Otherwise, if episodes span the full dataset from beginning to
                        # end_of_data with no max_steps, estimate episode length from the
                        # simulator length and the deterministic start index used by the env.
                        elif (
                            getattr(self.env, "start_policy", None) == "beginning"
                            and getattr(self.env, "end_policy", None) == "end_of_data"
                            and getattr(self.env, "max_steps", None) is None
                        ):
                            # Mirror TradingEnv.reset's computation of the deterministic
                            # start index (min_start_idx) used when start_policy == 'beginning'.
                            sim_len = int(getattr(self.env.simulator, "length", 0))
                            win_len = int(getattr(self.env, "window_length", 0))
                            min_start_idx = min(max(win_len - 1, 0), max(sim_len - 2, 0))
                            # Number of decision steps until the simulator runs out of bars.
                            episode_expected = max(int(sim_len - 1 - min_start_idx), 0)
                    except Exception:
                        episode_expected = None

                    if episode > 0 and episode_expected is not None and remaining_steps < episode_expected:
                        logger.info(
                            "Not enough remaining timesteps (%d) to start a new full episode (needs %d); finishing training to avoid truncated episode.",
                            remaining_steps,
                            episode_expected,
                        )
                        break

                    # Start a new episode. Do not pass seeds to reset; environment controls randomness.
                    obs, info = self.env.reset()

                    episode_reward = 0.0
                    episode_steps = 0
                    component_totals: Dict[str, float] = {}
                    episode_equity = [info.get("equity", info.get("equity_after", 0.0))]
                    episode_trade_log = []
                    episode_actions = []
                    done = False

                    # Determine per-episode total steps for the progress bar. Prefer
                    # an environment-provided value (info may include "max_episode_steps");
                    # fall back to the configured training.max_episode_steps if present.
                    episode_total: Optional[int] = None
                    try:
                        episode_total = info.get("max_episode_steps") if isinstance(info, dict) else None
                        if episode_total is None:
                            episode_total = self.config.get("training", {}).get("max_episode_steps")
                        if episode_total is not None:
                            episode_total = int(episode_total)
                            if episode_total <= 0:
                                episode_total = None
                    except Exception:
                        # Be conservative: if anything goes wrong, leave total unspecified.
                        episode_total = None

                    with progress_bar(
                        self.config,
                        desc=f"Episode {episode + 1}",
                        total=episode_total,
                        initial=0,
                        position=self.progress_position_offset + 1,
                        leave=False,
                        unit="step",
                        colour=episode_colour,
                    ) as episode_pbar:
                        while not done:

                            flat_obs = obs["flat"]
                            mask = obs["mask"]

                            # Agent owns exploration strategy; do not override epsilon from trainer.
                            action = self.agent.act(flat_obs, mask, training=True)
                            # Per-episode canonical action records are created after env.step
                            # where 'info' is available and global_step has been incremented.

                            next_obs, reward, terminated, truncated, info = self.env.step(action)
                            done = terminated or truncated

                            next_flat = next_obs["flat"]
                            next_mask = next_obs["mask"]

                            self.replay.add(flat_obs, action, reward, next_flat, done, mask, next_mask)

                            # Learn
                            loss_val = None
                            if global_step >= self.learn_start and self.replay.is_ready(self.batch_size):
                                if global_step % self.learn_frequency == 0:
                                    batch = self.replay.sample(self.batch_size)
                                    metrics = self.agent.update(batch)
                                    loss_val = metrics.get("loss", 0.0)
                                    total_loss_sum += float(loss_val)
                                    total_loss_count += 1
                                    self.reward_logger.log_loss(global_step, loss_val)
                                    if "q_mean" in metrics:
                                        q_window.append(float(metrics["q_mean"]))

                            # Epsilon update (use agent's internal decay which is updated in update())
                            episode_reward += reward
                            reward_window.append(float(reward))
                            episode_steps += 1
                            global_step += 1
                            training_pbar.update(1)
                            episode_pbar.update(1)


                            # Optional curriculum progression.
                            if self.curriculum.advance(global_step):
                                logger.info(
                                    "Curriculum advanced at step %d → %s",
                                    global_step,
                                    self.curriculum.get_current_stage_name(),
                                )

                            executed_action = int(info.get("executed_action", action))
                            if 0 <= executed_action < len(action_counts):
                                action_counts[executed_action] += 1

                            # Accumulate component rewards
                            rb = info.get("reward_breakdown", {})
                            for name, val in rb.items():
                                if isinstance(val, dict) and "weighted" in val:
                                    component_totals[name] = component_totals.get(name, 0.0) + val["weighted"]
                                elif isinstance(val, (int, float, np.floating)):
                                    component_totals[name] = component_totals.get(name, 0.0) + float(val)

                            episode_equity.append(info.get("equity_after", info.get("equity", episode_equity[-1])))
                            trade_event = build_trade_event(info, step=episode_steps)
                            if trade_event is not None:
                                episode_trade_log.append(trade_event)
                                training_trade_log.append(trade_event)
                                self.reward_logger.log_trade_event(global_step, episode + 1, trade_event)

                            # Log actions (global CSV)
                            self.reward_logger.log_action(global_step, info)

                            # Record per-episode action in the canonical schema so that the
                            # episode-level CSV matches the global actions_sequence format.
                            try:
                                episode_actions.append({
                                    "global_step": global_step,
                                    "timestamp": info.get("timestamp_decision", ""),
                                    "action_id": info.get("raw_action", ""),
                                    "action_name": info.get("action_name", ""),
                                    "was_legal": info.get("was_legal", ""),
                                    "executed_action_name": info.get("executed_action_name", info.get("action_name", "")),
                                    "direction_before": info.get("direction_before", ""),
                                    "direction_after": info.get("direction", ""),
                                    "total_lots_after": info.get("total_lots", ""),
                                })
                            except Exception:
                                logger.exception("Failed to record per-step action for episode %d", episode)

                            if (episode_steps % metrics_update_interval) == 0 or done:
                                common_metrics = {
                                    "ep": episode + 1,
                                    "eps": f"{self.agent.epsilon:.4f}",
                                    "rew": f"{episode_reward:.2f}",
                                }
                                if loss_val is not None:
                                    common_metrics["loss"] = f"{float(loss_val):.4f}"
                                set_bar_metrics(training_pbar, metrics=common_metrics, refresh=False)
                                set_bar_metrics(
                                    episode_pbar,
                                    metrics={
                                        "gstep": global_step,
                                        "reward": f"{episode_reward:.2f}",
                                        "eps": f"{self.agent.epsilon:.4f}",
                                    },
                                    refresh=False,
                                )

                            if self.evaluation_interval > 0 and global_step % self.evaluation_interval == 0:
                                self.reward_logger.log_evaluation_snapshot(
                                    global_step=global_step,
                                    episode=episode + 1,
                                    epsilon=float(self.agent.epsilon),
                                    replay_size=len(self.replay),
                                    equity=float(info.get("equity_after", info.get("equity", 0.0))),
                                    episode_reward=float(episode_reward),
                                )

                            if self.log_interval > 0 and global_step % self.log_interval == 0:
                                r_mean = float(np.mean(reward_window)) if reward_window else 0.0
                                r_std = float(np.std(reward_window)) if reward_window else 0.0
                                q_mean = float(np.mean(q_window)) if q_window else 0.0
                                q_std = float(np.std(q_window)) if q_window else 0.0

                                total_actions = int(np.sum(action_counts))
                                hold_ratio = (float(action_counts[0]) / total_actions) if total_actions > 0 else 0.0
                                trade_ratio = 1.0 - hold_ratio if total_actions > 0 else 0.0

                                action_dist = {}
                                if total_actions > 0:
                                    for action_id, cnt in enumerate(action_counts.tolist()):
                                        action_dist[ACTION_NAMES.get(action_id, str(action_id))] = float(cnt) / float(total_actions)

                                current_equity = float(info.get("equity_after", info.get("equity", 0.0)))
                                self.reward_logger.log_training_diagnostics(
                                    global_step=global_step,
                                    episode=episode + 1,
                                    reward_mean=r_mean,
                                    reward_std=r_std,
                                    epsilon=float(self.agent.epsilon),
                                    replay_size=len(self.replay),
                                    q_mean=q_mean,
                                    q_std=q_std,
                                    equity=current_equity,
                                    hold_ratio=hold_ratio,
                                    trade_ratio=trade_ratio,
                                    action_distribution=action_dist,
                                )

                                logger.info(
                                    "Diag step=%d ep=%d r_mean=%.5f r_std=%.5f eps=%.4f replay=%d q_mean=%.5f q_std=%.5f hold=%.2f trade=%.2f equity=%.2f",
                                    global_step,
                                    episode + 1,
                                    r_mean,
                                    r_std,
                                    float(self.agent.epsilon),
                                    len(self.replay),
                                    q_mean,
                                    q_std,
                                    hold_ratio,
                                    trade_ratio,
                                    current_equity,
                                )

                            obs = next_obs

                    # Episode complete
                    episode += 1
                    best_episode_reward = max(best_episode_reward, episode_reward)

                    episode_metrics = compute_metrics(
                        np.asarray(episode_equity, dtype=np.float64),
                        trade_log=episode_trade_log,
                        periods_per_year=self.config.get("evaluation", {}).get("periods_per_year", 6048),
                        risk_free_rate=self.config.get("evaluation", {}).get("risk_free_rate", 0.0),
                    )
                    last_episode_metrics = episode_metrics
                    last_episode_equity = np.asarray(episode_equity, dtype=np.float64)

                    # Accumulate this episode's equity into the full training equity curve.
                    # Avoid duplicating the boundary point between consecutive episodes by
                    # skipping the first value of subsequent episodes (it's the same as the
                    # previous episode's last point).
                    try:
                        if last_episode_equity is not None and len(last_episode_equity) > 0:
                            if not training_equity:
                                training_equity.extend(list(last_episode_equity))
                            else:
                                training_equity.extend(list(last_episode_equity[1:]))
                    except Exception:
                        logger.exception("Failed to accumulate episode equity into training equity")

                    self.reward_logger.log_episode_reward(
                        global_step, episode, episode_reward, component_totals,
                    )
                    self.reward_logger.log_episode_metrics(global_step, episode, episode_metrics)

                    # Save per-episode metrics and action sequence
                    try:
                        metrics_train_dir = Path(self.dirs.get("metrics_train", self.run_dir / "metrics" / "train"))
                        episodes_dir = metrics_train_dir / "episodes"
                        episodes_dir.mkdir(parents=True, exist_ok=True)

                        # Equity & drawdown curves
                        if last_episode_equity is not None and len(last_episode_equity) > 0:
                            save_equity_curve(last_episode_equity, episodes_dir / f"episode_{episode}_equity_curve.csv")
                            save_drawdown_curve(last_episode_equity, episodes_dir / f"episode_{episode}_drawdown_curve.csv")

                        # Action sequence (per-episode) — save using canonical schema so
                        # it matches the global `actions_sequence.csv` format.
                        actions_columns = [
                            "global_step", "timestamp", "action_id", "action_name",
                            "was_legal", "executed_action_name", "direction_before",
                            "direction_after", "total_lots_after",
                        ]
                        actions_df = pd.DataFrame(episode_actions)
                        # Ensure consistent column order and fill missing values with empty string
                        actions_df = actions_df.reindex(columns=actions_columns)
                        actions_df.fillna("", inplace=True)
                        actions_df.to_csv(episodes_dir / f"episode_{episode}_action_sequence.csv", index=False)

                        trade_columns = [
                            "step",
                            "pnl",
                            "direction_before",
                            "direction",
                            "forced_liquidation",
                            "pyramid_steps",
                            "martingale_steps",
                            "notional",
                        ]
                        episode_trade_df = pd.DataFrame(episode_trade_log).reindex(columns=trade_columns)
                        episode_trade_df.fillna(0.0, inplace=True)
                        episode_trade_df.to_csv(episodes_dir / f"episode_{episode}_trade_log.csv", index=False)
                    except Exception:
                        logger.exception("Failed to save per-episode metrics for episode %d", episode)

                    # Save checkpoints on episode boundaries only to avoid unrecoverable
                    # mid-episode resume states (environment state is not serialized).
                    if self.checkpoint_interval > 0 and (global_step - last_checkpoint_step) >= self.checkpoint_interval:
                        self._save_checkpoint(global_step, episode)
                        last_checkpoint_step = global_step

                    if episode % max(1, self.log_interval // max(episode_steps, 1)) == 0 or episode <= 5:
                        logger.info(
                            "Episode %d | step %d/%d | reward=%.4f | eps=%.4f | losses=%d",
                            episode, global_step, self.total_timesteps,
                            episode_reward, self.agent.epsilon, total_loss_count,
                        )

        elapsed = time.time() - t0
        logger.info("Training completed in %.1fs — %d episodes, %d steps", elapsed, episode, global_step)

        # Save final checkpoint + model
        self._save_checkpoint(global_step, episode)
        models_dir = self.dirs.get("models", self.run_dir / "models")
        self.ckpt_mgr.save_final_model(self.agent, models_dir)
        self.reward_logger.close()

        tables_dir = self.dirs.get("tables_train", self.run_dir / "tables" / "train")
        metrics_dir = self.dirs.get("metrics_train", self.run_dir / "metrics" / "train")

        training_equity_arr = None
        if training_equity:
            training_equity_arr = np.asarray(training_equity, dtype=np.float64)
        elif last_episode_equity is not None and len(last_episode_equity) > 0:
            training_equity_arr = np.asarray(last_episode_equity, dtype=np.float64)

        if training_equity_arr is not None and training_equity_arr.size > 0:
            try:
                save_equity_curve(training_equity_arr, metrics_dir / "equity_curve.csv")
                save_drawdown_curve(training_equity_arr, metrics_dir / "drawdown_curve.csv")
            except Exception:
                logger.exception("Failed to save full training equity/drawdown curves")

        full_training_metrics: Dict[str, float] = {}
        if training_equity_arr is not None and training_equity_arr.size > 0:
            full_training_metrics = compute_metrics(
                training_equity_arr,
                trade_log=training_trade_log,
                periods_per_year=self.config.get("evaluation", {}).get("periods_per_year", 6048),
                risk_free_rate=self.config.get("evaluation", {}).get("risk_free_rate", 0.0),
            )

        if full_training_metrics:
            save_performance_summary(full_training_metrics, tables_dir / "performance_summary.csv")
            save_risk_metrics(full_training_metrics, tables_dir / "risk_metrics.csv")
            save_performance_summary(full_training_metrics, metrics_dir / "full_training_metrics.csv")
            save_risk_metrics(full_training_metrics, metrics_dir / "full_training_risk_metrics.csv")

        save_trade_log(training_trade_log, metrics_dir / "trade_log.csv")

        if last_episode_metrics:
            save_performance_summary(last_episode_metrics, tables_dir / "performance_summary_last_episode.csv")
            save_risk_metrics(last_episode_metrics, tables_dir / "risk_metrics_last_episode.csv")

        plot_equity_source = None
        if training_equity_arr is not None and training_equity_arr.size > 0:
            plot_equity_source = training_equity_arr
        elif last_episode_equity is not None and len(last_episode_equity) > 0:
            plot_equity_source = last_episode_equity

        if plot_equity_source is not None and len(plot_equity_source) > 0:
            figures_dir = self.dirs.get("figures_train", self.run_dir / "figures" / "train")
            plot_equity_curve(
                plot_equity_source,
                figures_dir / f"equity_curve.{self.figure_format}",
                title=f"{self.pair} Training Equity",
                fmt=self.figure_format,
            )
            plot_drawdown_curve(
                plot_equity_source,
                figures_dir / f"drawdown_curve.{self.figure_format}",
                title=f"{self.pair} Training Drawdown",
                fmt=self.figure_format,
            )

        summary = {
            "pair": self.pair,
            "agent": self.config.get("agent", {}).get("name", ""),
            "total_episodes": episode,
            "total_steps": global_step,
            "best_episode_reward": best_episode_reward,
            "mean_loss": (float(total_loss_sum) / float(total_loss_count)) if total_loss_count > 0 else 0.0,
            "elapsed_seconds": elapsed,
            "final_cumulative_return": float(full_training_metrics.get("cumulative_return", 0.0)) if full_training_metrics else 0.0,
            "final_sharpe_ratio": float(full_training_metrics.get("sharpe_ratio", 0.0)) if full_training_metrics else 0.0,
            "final_max_drawdown": float(full_training_metrics.get("max_drawdown", 0.0)) if full_training_metrics else 0.0,
        }
        write_log_file(
            self.train_log_path,
            [
                f"training_complete episodes={episode}",
                f"total_steps={global_step}",
                f"best_episode_reward={best_episode_reward:.6f}",
                f"mean_loss={summary['mean_loss']:.6f}",
                f"elapsed_seconds={elapsed:.3f}",
            ],
            mode="a",
        )
        return summary