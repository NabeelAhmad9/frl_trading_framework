"""Random policy benchmark — sample legal actions with fixed seed."""

import numpy as np
from typing import Any, Dict

from src.benchmarks.base_benchmark import BaseBenchmark


class RandomPolicyBenchmark(BaseBenchmark):
    """Sample uniformly from legal actions."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        seed = config.get("benchmark", {}).get("random_seed", 42)
        self._rng = np.random.RandomState(seed)

    def reset(self) -> None:
        pass  # Preserve RNG state across episodes for reproducibility

    def act(self, observation: Dict[str, np.ndarray], info: Dict[str, Any], step: int) -> int:
        mask = observation["mask"]
        legal = np.where(mask > 0)[0]
        if len(legal) == 0:
            return 0  # HOLD fallback
        return int(self._rng.choice(legal))
