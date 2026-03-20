"""Reward logger — persist total and component rewards as CSVs."""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RewardLogger:
    """Log reward curves and loss curves to CSV files with stable schema."""

    def __init__(
        self,
        metrics_dir: Path,
        log_components: bool = True,
        component_interval: int = 1000,
        append_mode: bool = False,
        flush_interval: int = 128,
    ):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.log_components = log_components
        self.component_interval = max(component_interval, 1)
        self.append_mode = bool(append_mode)
        self.flush_interval = max(int(flush_interval), 1)

        self._reward_path = self.metrics_dir / "reward_curve.csv"
        self._loss_path = self.metrics_dir / "loss_curve.csv"
        self._action_path = self.metrics_dir / "actions_sequence.csv"
        self._trade_path = self.metrics_dir / "trade_log.csv"
        self._diagnostics_path = self.metrics_dir / "training_diagnostics.csv"
        self._eval_snapshot_path = self.metrics_dir / "evaluation_snapshots.csv"
        self._episode_metric_paths = {
            "cumulative_return": self.metrics_dir / "cumulative_return.csv",
            "sharpe_ratio": self.metrics_dir / "sharpe_ratio.csv",
            "max_drawdown": self.metrics_dir / "max_drawdown.csv",
            "turnover": self.metrics_dir / "turnover.csv",
        }

        self._reward_columns: Optional[List[str]] = None
        self._reward_writer = None
        self._reward_file = None
        self._reward_rows: List[Dict[str, Any]] = []

        self._loss_file = None
        self._loss_writer = None

        self._action_file = None
        self._action_writer = None

        self._trade_file = None
        self._trade_writer = None

        self._diagnostics_file = None
        self._diagnostics_writer = None

        self._eval_snapshot_file = None
        self._eval_snapshot_writer = None

        self._episode_metric_files: Dict[str, Any] = {}
        self._episode_metric_writers: Dict[str, Any] = {}
        self._pending_flush_counts: Dict[str, int] = {
            "reward": 0,
            "loss": 0,
            "action": 0,
            "trade": 0,
            "diagnostics": 0,
            "eval": 0,
        }

    def _flush_if_needed(self, key: str, fh: Any, force: bool = False) -> None:
        if fh is None:
            return
        if force:
            fh.flush()
            self._pending_flush_counts[key] = 0
            return
        self._pending_flush_counts[key] = self._pending_flush_counts.get(key, 0) + 1
        if self._pending_flush_counts[key] >= self.flush_interval:
            fh.flush()
            self._pending_flush_counts[key] = 0

    def _ensure_reward_file(self, component_names: List[str]) -> None:
        component_names = sorted(component_names)
        base_cols = ["global_step", "episode", "reward_total"]
        desired_cols = base_cols + ([f"reward_{name}" for name in component_names] if self.log_components else [])

        if self._reward_file is None:
            existing_rows: List[Dict[str, Any]] = []
            existing_cols: List[str] = []
            if self.append_mode and self._reward_path.exists() and self._reward_path.stat().st_size > 0:
                with open(self._reward_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    existing_cols = list(reader.fieldnames or [])
                    existing_rows = [dict(row) for row in reader]

            if existing_cols:
                self._reward_columns = list(existing_cols)
            else:
                self._reward_columns = list(desired_cols)

            missing_cols = [c for c in desired_cols if c not in self._reward_columns]
            if missing_cols:
                self._reward_columns.extend(missing_cols)

            needs_rewrite = bool(existing_rows and missing_cols)
            if needs_rewrite:
                self._reward_file = open(self._reward_path, "w", newline="", encoding="utf-8")
                self._reward_writer = csv.DictWriter(self._reward_file, fieldnames=self._reward_columns, extrasaction="ignore")
                self._reward_writer.writeheader()
                for old_row in existing_rows:
                    self._reward_writer.writerow(old_row)
                self._flush_if_needed("reward", self._reward_file, force=True)
            else:
                mode = "a" if (self.append_mode and self._reward_path.exists()) else "w"
                self._reward_file = open(self._reward_path, mode, newline="", encoding="utf-8")
                self._reward_writer = csv.DictWriter(self._reward_file, fieldnames=self._reward_columns, extrasaction="ignore")
                if mode == "w" or self._reward_path.stat().st_size == 0:
                    self._reward_writer.writeheader()

            self._reward_rows = list(existing_rows)
            return

        # Expand schema when new reward components appear later in training.
        assert self._reward_columns is not None
        missing_cols = [c for c in desired_cols if c not in self._reward_columns]
        if not missing_cols:
            return

        self._reward_columns.extend(missing_cols)
        self._reward_file.close()
        self._reward_file = open(self._reward_path, "w", newline="", encoding="utf-8")
        self._reward_writer = csv.DictWriter(self._reward_file, fieldnames=self._reward_columns, extrasaction="ignore")
        self._reward_writer.writeheader()
        for old_row in self._reward_rows:
            self._reward_writer.writerow(old_row)
        self._flush_if_needed("reward", self._reward_file, force=True)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(out):
            return default
        return out

    def _ensure_loss_file(self) -> None:
        if self._loss_file is not None:
            return
        mode = "a" if (self.append_mode and self._loss_path.exists()) else "w"
        self._loss_file = open(self._loss_path, mode, newline="", encoding="utf-8")
        self._loss_writer = csv.DictWriter(self._loss_file, fieldnames=["global_step", "td_loss"])
        if mode == "w" or self._loss_path.stat().st_size == 0:
            self._loss_writer.writeheader()

    def _ensure_action_file(self) -> None:
        if self._action_file is not None:
            return
        cols = [
            "global_step", "timestamp", "action_id", "action_name",
            "was_legal", "executed_action_name", "direction_before",
            "direction_after", "total_lots_after",
        ]
        mode = "a" if (self.append_mode and self._action_path.exists()) else "w"
        self._action_file = open(self._action_path, mode, newline="", encoding="utf-8")
        self._action_writer = csv.DictWriter(self._action_file, fieldnames=cols)
        if mode == "w" or self._action_path.stat().st_size == 0:
            self._action_writer.writeheader()

    def _ensure_trade_file(self) -> None:
        if self._trade_file is not None:
            return
        cols = [
            "global_step",
            "episode",
            "step",
            "pnl",
            "direction_before",
            "direction",
            "forced_liquidation",
            "pyramid_steps",
            "martingale_steps",
            "notional",
        ]
        mode = "a" if (self.append_mode and self._trade_path.exists()) else "w"
        self._trade_file = open(self._trade_path, mode, newline="", encoding="utf-8")
        self._trade_writer = csv.DictWriter(self._trade_file, fieldnames=cols)
        if mode == "w" or self._trade_path.stat().st_size == 0:
            self._trade_writer.writeheader()

    def _ensure_diagnostics_file(self) -> None:
        if self._diagnostics_file is not None:
            return

        cols = [
            "global_step", "episode", "reward_mean", "reward_std", "epsilon",
            "replay_size", "q_mean", "q_std", "equity", "hold_ratio",
            "trade_ratio", "action_distribution_json",
        ]
        mode = "a" if (self.append_mode and self._diagnostics_path.exists()) else "w"
        self._diagnostics_file = open(self._diagnostics_path, mode, newline="", encoding="utf-8")
        self._diagnostics_writer = csv.DictWriter(self._diagnostics_file, fieldnames=cols)
        if mode == "w" or self._diagnostics_path.stat().st_size == 0:
            self._diagnostics_writer.writeheader()

    def _ensure_eval_snapshot_file(self) -> None:
        if self._eval_snapshot_file is not None:
            return

        cols = ["global_step", "episode", "epsilon", "replay_size", "equity", "episode_reward"]
        mode = "a" if (self.append_mode and self._eval_snapshot_path.exists()) else "w"
        self._eval_snapshot_file = open(self._eval_snapshot_path, mode, newline="", encoding="utf-8")
        self._eval_snapshot_writer = csv.DictWriter(self._eval_snapshot_file, fieldnames=cols)
        if mode == "w" or self._eval_snapshot_path.stat().st_size == 0:
            self._eval_snapshot_writer.writeheader()

    def _ensure_episode_metric_file(self, metric_name: str) -> None:
        if metric_name in self._episode_metric_files:
            return
        path = self._episode_metric_paths[metric_name]
        mode = "a" if (self.append_mode and path.exists()) else "w"
        fh = open(path, mode, newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=["global_step", "episode", "value"])
        if mode == "w" or path.stat().st_size == 0:
            writer.writeheader()
        self._episode_metric_files[metric_name] = fh
        self._episode_metric_writers[metric_name] = writer

    def log_episode_reward(
        self,
        global_step: int,
        episode: int,
        total_reward: float,
        component_breakdown: Optional[Dict[str, float]] = None,
    ) -> None:
        comp_names = sorted(component_breakdown.keys()) if component_breakdown else []
        self._ensure_reward_file(comp_names)

        row = {
            "global_step": int(global_step),
            "episode": int(episode),
            "reward_total": self._safe_float(total_reward),
        }
        if component_breakdown and self.log_components:
            for name, val in component_breakdown.items():
                row[f"reward_{name}"] = self._safe_float(val)
        self._reward_writer.writerow(row)
        self._reward_rows.append(dict(row))
        self._flush_if_needed("reward", self._reward_file)

    def log_loss(self, global_step: int, loss: float) -> None:
        self._ensure_loss_file()
        self._loss_writer.writerow({"global_step": int(global_step), "td_loss": self._safe_float(loss)})
        self._flush_if_needed("loss", self._loss_file)

    def log_action(self, global_step: int, info: Dict[str, Any]) -> None:
        self._ensure_action_file()
        self._action_writer.writerow({
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
        self._flush_if_needed("action", self._action_file)

    def log_trade_event(self, global_step: int, episode: int, trade_event: Dict[str, Any]) -> None:
        self._ensure_trade_file()
        self._trade_writer.writerow({
            "global_step": int(global_step),
            "episode": int(episode),
            "step": int(trade_event.get("step", 0)),
            "pnl": self._safe_float(trade_event.get("pnl", 0.0)),
            "direction_before": int(trade_event.get("direction_before", 0)),
            "direction": int(trade_event.get("direction", 0)),
            "forced_liquidation": bool(trade_event.get("forced_liquidation", False)),
            "pyramid_steps": int(trade_event.get("pyramid_steps", 0)),
            "martingale_steps": int(trade_event.get("martingale_steps", 0)),
            "notional": self._safe_float(trade_event.get("notional", 0.0)),
        })
        self._flush_if_needed("trade", self._trade_file)

    def log_episode_metrics(self, global_step: int, episode: int, metrics: Dict[str, float]) -> None:
        for metric_name in self._episode_metric_paths:
            if metric_name not in metrics:
                continue
            self._ensure_episode_metric_file(metric_name)
            self._episode_metric_writers[metric_name].writerow({
                "global_step": int(global_step),
                "episode": int(episode),
                "value": self._safe_float(metrics[metric_name]),
            })
            self._episode_metric_files[metric_name].flush()

    def log_training_diagnostics(
        self,
        global_step: int,
        episode: int,
        reward_mean: float,
        reward_std: float,
        epsilon: float,
        replay_size: int,
        q_mean: float,
        q_std: float,
        equity: float,
        hold_ratio: float,
        trade_ratio: float,
        action_distribution: Optional[Dict[str, float]] = None,
    ) -> None:
        self._ensure_diagnostics_file()
        self._diagnostics_writer.writerow({
            "global_step": int(global_step),
            "episode": int(episode),
            "reward_mean": self._safe_float(reward_mean),
            "reward_std": self._safe_float(reward_std),
            "epsilon": self._safe_float(epsilon),
            "replay_size": int(replay_size),
            "q_mean": self._safe_float(q_mean),
            "q_std": self._safe_float(q_std),
            "equity": self._safe_float(equity),
            "hold_ratio": self._safe_float(hold_ratio),
            "trade_ratio": self._safe_float(trade_ratio),
            "action_distribution_json": json.dumps(action_distribution or {}, sort_keys=True),
        })
        self._flush_if_needed("diagnostics", self._diagnostics_file)

    def log_evaluation_snapshot(
        self,
        global_step: int,
        episode: int,
        epsilon: float,
        replay_size: int,
        equity: float,
        episode_reward: float,
    ) -> None:
        self._ensure_eval_snapshot_file()
        self._eval_snapshot_writer.writerow({
            "global_step": int(global_step),
            "episode": int(episode),
            "epsilon": self._safe_float(epsilon),
            "replay_size": int(replay_size),
            "equity": self._safe_float(equity),
            "episode_reward": self._safe_float(episode_reward),
        })
        self._flush_if_needed("eval", self._eval_snapshot_file)

    def close(self) -> None:
        self._flush_if_needed("reward", self._reward_file, force=True)
        self._flush_if_needed("loss", self._loss_file, force=True)
        self._flush_if_needed("action", self._action_file, force=True)
        self._flush_if_needed("trade", self._trade_file, force=True)
        self._flush_if_needed("diagnostics", self._diagnostics_file, force=True)
        self._flush_if_needed("eval", self._eval_snapshot_file, force=True)
        for f in [
            self._reward_file,
            self._loss_file,
            self._action_file,
            self._trade_file,
            self._diagnostics_file,
            self._eval_snapshot_file,
            *self._episode_metric_files.values(),
        ]:
            if f is not None:
                f.close()
        self._reward_file = None
        self._loss_file = None
        self._action_file = None
        self._trade_file = None
        self._diagnostics_file = None
        self._eval_snapshot_file = None
        self._trade_writer = None
        self._episode_metric_files = {}
        self._episode_metric_writers = {}
