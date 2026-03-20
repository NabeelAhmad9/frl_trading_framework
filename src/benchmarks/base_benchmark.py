"""Base benchmark — common interface for benchmark policies."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseBenchmark(ABC):
    """Base class for all benchmark strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        bench_cfg = config.get("benchmark", {})
        self.name = bench_cfg.get("name", "unknown")
        self.description = bench_cfg.get("description", "")

    @abstractmethod
    def act(self, observation: Dict[str, np.ndarray], info: Dict[str, Any], step: int) -> int:
        """Return an action given current observation and info.

        Parameters
        ----------
        observation : dict
            Observation dict with keys ``market``, ``portfolio``, ``mask``, ``flat``.
        info : dict
            Info dict from env.
        step : int
            Current step number.
        """

    def reset(self) -> None:
        """Reset any internal state for a new episode."""
        pass

    def metadata(self) -> Dict[str, str]:
        return {"name": self.name, "description": self.description}
