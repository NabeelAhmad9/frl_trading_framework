"""Base agent interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseAgent(ABC):
    """Abstract base for all RL agents."""

    @abstractmethod
    def act(self, observation: np.ndarray, mask: np.ndarray, training: bool = True) -> int:
        """Select an action."""

    @abstractmethod
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update from a replay batch. Returns loss info."""

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state."""

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state."""

    @abstractmethod
    def train_mode(self) -> None:
        """Set to training mode."""

    @abstractmethod
    def eval_mode(self) -> None:
        """Set to evaluation mode."""
