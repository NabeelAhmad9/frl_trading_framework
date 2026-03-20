"""Mean-reversion benchmark — contrarian strategy based on z-score."""

import numpy as np
from collections import deque
from typing import Any, Dict

from src.benchmarks.base_benchmark import BaseBenchmark
from src.environment.action_space import Action


class MeanReversionBenchmark(BaseBenchmark):
    """Trade against overextension using rolling z-score of close prices."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        bench_cfg = config.get("benchmark", {})
        self.lookback = bench_cfg.get("lookback_window", 20)
        self.entry_z = bench_cfg.get("entry_z_threshold", 2.0)
        self.exit_z = bench_cfg.get("exit_z_threshold", 0.5)
        self._prices: deque = deque(maxlen=self.lookback)

    def reset(self) -> None:
        self._prices.clear()

    def act(self, observation: Dict[str, np.ndarray], info: Dict[str, Any], step: int) -> int:
        mask = observation["mask"]
        market = observation.get("market", None)

        # Extract latest close from market window (last row, 4th column = close)
        if market is not None and len(market.shape) == 2 and market.shape[1] >= 4:
            close = float(market[-1, 3])
        else:
            return int(Action.HOLD)

        self._prices.append(close)

        if len(self._prices) < self.lookback:
            return int(Action.HOLD)

        arr = np.array(self._prices)
        mean = arr.mean()
        std = arr.std()
        if std < 1e-12:
            return int(Action.HOLD)
        z = (close - mean) / std

        direction = info.get("direction", 0)

        # Short if z > entry_z (overextended up)
        if z > self.entry_z and direction <= 0:
            if direction == 0 and mask[Action.OPEN_SHORT] > 0:
                return int(Action.OPEN_SHORT)

        # Long if z < -entry_z (overextended down)
        if z < -self.entry_z and direction >= 0:
            if direction == 0 and mask[Action.OPEN_LONG] > 0:
                return int(Action.OPEN_LONG)

        # Close if z reverts to within exit threshold
        if direction != 0 and abs(z) < self.exit_z:
            if mask[Action.CLOSE_POSITION] > 0:
                return int(Action.CLOSE_POSITION)

        return int(Action.HOLD)
