"""Resume manager — detect and restore training state."""

import json
from typing import Any, Dict, Tuple

from src.training.checkpoint_manager import CheckpointManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResumeManager:
    """Detect resumable state and restore agent/counters."""

    def __init__(self, ckpt_manager: CheckpointManager):
        self.ckpt_manager = ckpt_manager

    def can_resume(self) -> bool:
        return (
            self.ckpt_manager.has_checkpoint()
            and self.ckpt_manager.has_training_state()
        )

    def restore(self, agent) -> Tuple[int, int, Dict[str, Any]]:
        """Restore agent weights and return (global_step, episode, state_dict).

        Raises FileNotFoundError if no resumable state exists.
        """
        if not self.can_resume():
            raise FileNotFoundError("No resumable checkpoint found")

        ckpt_path = self.ckpt_manager.latest_checkpoint_path()
        state_path = self.ckpt_manager.training_state_path()

        agent.load_checkpoint(str(ckpt_path))

        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        runtime_state = self.ckpt_manager.load_runtime_state()
        if runtime_state is not None:
            state["runtime_state"] = runtime_state

        global_step = max(int(state.get("global_step", 0)), 0)
        episode = max(int(state.get("episode", 0)), 0)

        logger.info(
            "Resumed from step=%d episode=%d (checkpoint: %s)",
            global_step, episode, ckpt_path,
        )
        return global_step, episode, state
