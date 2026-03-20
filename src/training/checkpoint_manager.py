"""Checkpoint manager — save and load training checkpoints."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Save canonical checkpoints and training state JSON."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        agent,
        global_step: int,
        episode: int,
        epsilon: float,
        config: dict,
        extra: Optional[Dict[str, Any]] = None,
        runtime_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model checkpoint, training_state.json, and optional runtime_state.pt."""
        ckpt_path = self.checkpoint_dir / "checkpoint_latest.pt"
        tmp_ckpt_path = self.checkpoint_dir / "checkpoint_latest.pt.tmp"
        agent.save_checkpoint(str(tmp_ckpt_path))
        os.replace(tmp_ckpt_path, ckpt_path)

        state = {
            "run_id": config.get("meta", {}).get("run_id", ""),
            "pair": config.get("data", {}).get("pair", ""),
            "agent": config.get("agent", {}).get("name", ""),
            "experiment": config.get("experiment", {}).get("name", ""),
            "variant": config.get("experiment", {}).get("variant", ""),
            "global_step": global_step,
            "episode": episode,
            "epsilon": epsilon,
            "random_seed": config.get("training", {}).get("random_seed", 42),
            "checkpoint_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            state.update(extra)

        state_path = self.checkpoint_dir / "training_state.json"
        tmp_state_path = self.checkpoint_dir / "training_state.json.tmp"
        with open(tmp_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_state_path, state_path)

        if runtime_state is not None:
            runtime_path = self.runtime_state_path()
            tmp_runtime_path = self.checkpoint_dir / "runtime_state.pt.tmp"
            torch.save(runtime_state, tmp_runtime_path)
            os.replace(tmp_runtime_path, runtime_path)

        logger.info("Checkpoint saved at step %d → %s", global_step, ckpt_path)
        return ckpt_path

    def save_final_model(self, agent, models_dir: Path) -> Path:
        """Save model_final.pt."""
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / "model_final.pt"
        agent.save_checkpoint(str(path))
        logger.info("Final model saved → %s", path)
        return path

    def latest_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "checkpoint_latest.pt"

    def training_state_path(self) -> Path:
        return self.checkpoint_dir / "training_state.json"

    def runtime_state_path(self) -> Path:
        return self.checkpoint_dir / "runtime_state.pt"

    def has_checkpoint(self) -> bool:
        return self.latest_checkpoint_path().exists()

    def has_training_state(self) -> bool:
        return self.training_state_path().exists()

    def load_runtime_state(self) -> Optional[Dict[str, Any]]:
        path = self.runtime_state_path()
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=False)
