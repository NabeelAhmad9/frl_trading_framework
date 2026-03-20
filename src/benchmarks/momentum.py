"""Momentum benchmark — trend-following heuristic using recent returns."""

import numpy as np
from collections import deque
from typing import Any, Dict

from src.benchmarks.base_benchmark import BaseBenchmark
from src.environment.action_space import Action


class MomentumBenchmark(BaseBenchmark):
    """Follow the recent trend — go long if cumulative return > threshold, short otherwise."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        bench_cfg = config.get("benchmark", {})
        self.lookback = bench_cfg.get("lookback_window", 20)
        self.threshold = bench_cfg.get("entry_threshold", 0.0)
        self._prices: deque = deque(maxlen=self.lookback + 1)

    def reset(self) -> None:
        self._prices.clear()

    def act(self, observation: Dict[str, np.ndarray], info: Dict[str, Any], step: int) -> int:
        mask = observation["mask"]
        market = observation.get("market", None)

        if market is not None and len(market.shape) == 2 and market.shape[1] >= 4:
            close = float(market[-1, 3])
        else:
            return int(Action.HOLD)

        self._prices.append(close)

        if len(self._prices) < self.lookback + 1:
            return int(Action.HOLD)

        arr = np.array(self._prices)
        ret = (arr[-1] / max(arr[0], 1e-12)) - 1.0

        direction = info.get("direction", 0)

        if ret > self.threshold:
            # Bullish momentum
            if direction <= 0:
                if direction < 0 and mask[Action.CLOSE_POSITION] > 0:
                    return int(Action.CLOSE_POSITION)
                if direction == 0 and mask[Action.OPEN_LONG] > 0:
                    return int(Action.OPEN_LONG)
        elif ret < -self.threshold:
            # Bearish momentum
            if direction >= 0:
                if direction > 0 and mask[Action.CLOSE_POSITION] > 0:
                    return int(Action.CLOSE_POSITION)
                if direction == 0 and mask[Action.OPEN_SHORT] > 0:
                    return int(Action.OPEN_SHORT)

        return int(Action.HOLD)
