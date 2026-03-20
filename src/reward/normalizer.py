"""Reward normalizer — clipping and optional running normalization."""

from typing import Any, Dict
import math


class RewardNormalizer:
    """Normalize reward via clipping (or future running normalization)."""

    def __init__(self, config: Dict[str, Any]):
        norm_cfg = config.get("reward", config).get("normalization", {})
        self.method = norm_cfg.get("method", "clip_only")
        self.clip_min = norm_cfg.get("clip_min", -10.0)
        self.clip_max = norm_cfg.get("clip_max", 10.0)
        self.tanh_scale = max(float(norm_cfg.get("tanh_scale", 1.0)), 1e-8)

    def normalize(self, reward: float) -> float:
        """Handle possible nan/inf and apply clipping."""
        # Sanitize input
        if reward is None:
            return 0.0

        reward = float(reward)
        if not math.isfinite(reward):
            return 0.0

        if self.method == "tanh":
            reward = math.tanh(reward / self.tanh_scale)

        if self.method == "clip_only":
            return max(self.clip_min, min(self.clip_max, reward))

        # Default/fallback: still enforce configured clipping bounds.
        return max(self.clip_min, min(self.clip_max, reward))
