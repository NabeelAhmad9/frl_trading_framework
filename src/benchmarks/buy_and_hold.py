"""Buy-and-hold benchmark — open long at first opportunity, hold, close at end."""

import numpy as np
from typing import Any, Dict

from src.benchmarks.base_benchmark import BaseBenchmark
from src.environment.action_space import Action


class BuyAndHoldBenchmark(BaseBenchmark):
    """Open long once, hold indefinitely."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._opened = False

    def reset(self) -> None:
        self._opened = False

    def act(self, observation: Dict[str, np.ndarray], info: Dict[str, Any], step: int) -> int:
        mask = observation["mask"]
        direction = info.get("direction", 0)

        if not self._opened and direction == 0:
            # Try to open long
            if mask[Action.OPEN_LONG] > 0:
                self._opened = True
                return int(Action.OPEN_LONG)

        # HOLD
        return int(Action.HOLD)
