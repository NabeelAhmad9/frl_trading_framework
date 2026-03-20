"""Curriculum scheduler — optional difficulty progression (disabled by default)."""

from typing import Any, Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CurriculumScheduler:
    """Optional step-based curriculum scheduler.

    Supports either:
      - training.curriculum.stages
      - training.curriculum.milestones

    with each stage carrying a step threshold and a difficulty/update dict.
    """

    def __init__(self, config: dict):
        cur_cfg = config.get("training", {}).get("curriculum", {})
        self.enabled = bool(cur_cfg.get("enabled", False))

        raw_stages = cur_cfg.get("stages", cur_cfg.get("milestones", [])) or []
        self._stages: List[Dict[str, Any]] = []

        for idx, stage in enumerate(raw_stages):
            if not isinstance(stage, dict):
                continue

            step_val = stage.get("step", stage.get("start_step", stage.get("global_step", None)))
            if step_val is None:
                continue

            try:
                step = max(int(step_val), 0)
            except (TypeError, ValueError):
                continue

            difficulty = stage.get("difficulty", stage.get("params", stage.get("updates", {})))
            if not isinstance(difficulty, dict):
                difficulty = {}

            self._stages.append(
                {
                    "step": step,
                    "name": str(stage.get("name", f"stage_{idx}")),
                    "difficulty": dict(difficulty),
                }
            )

        self._stages.sort(key=lambda x: x["step"])

        # Ensure stage-0 baseline exists so current difficulty is always well-defined.
        if self._stages and self._stages[0]["step"] > 0:
            self._stages.insert(0, {"step": 0, "name": "baseline", "difficulty": {}})

        if self.enabled and not self._stages:
            logger.warning("Curriculum is enabled but no valid stages were configured; disabling curriculum.")
            self.enabled = False

        self._stage_idx = 0

    def should_advance(self, global_step: int) -> bool:
        """Return True if difficulty should increase at this step."""
        if not self.enabled or not self._stages:
            return False
        if self._stage_idx >= len(self._stages) - 1:
            return False
        next_stage_step = self._stages[self._stage_idx + 1]["step"]
        return int(global_step) >= int(next_stage_step)

    def advance(self, global_step: int) -> bool:
        """Advance stage if milestone(s) were reached. Returns True if changed."""
        advanced = False
        while self.should_advance(global_step):
            self._stage_idx += 1
            advanced = True
        return advanced

    def get_current_difficulty(self) -> dict:
        """Return current difficulty parameters (no-op when disabled)."""
        if not self.enabled or not self._stages:
            return {}
        return dict(self._stages[self._stage_idx]["difficulty"])

    def get_current_stage_name(self) -> str:
        if not self.enabled or not self._stages:
            return "disabled"
        return str(self._stages[self._stage_idx]["name"])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "stage_idx": self._stage_idx,
            "stages": self._stages,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        saved_enabled = bool(state.get("enabled", self.enabled))
        if not self.enabled and saved_enabled:
            # Respect current config when resuming; don't silently enable if disabled now.
            logger.warning("Ignoring enabled=True from checkpoint because current config has curriculum disabled.")

        stage_idx = int(state.get("stage_idx", self._stage_idx))
        if self._stages:
            self._stage_idx = max(0, min(stage_idx, len(self._stages) - 1))
        else:
            self._stage_idx = 0
