"""Seed management — set seeds across Python, NumPy, PyTorch, and env."""

import os
import random
from typing import Any, Dict

import numpy as np
import torch

from src.utils.seed import set_global_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def init_seeds(config: dict) -> int:
    """Set all global seeds from config and return the seed value."""
    seed = int(config.get("training", {}).get("random_seed", 42))
    deterministic = config.get("training", {}).get("deterministic_torch", False)
    set_global_seed(seed, deterministic_torch=deterministic)

    # Use deterministic kernel paths when requested (best effort).
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            logger.exception("Failed to enable torch deterministic algorithms")

    logger.info("Seed manager initialised seed=%d", seed)
    return seed


def capture_rng_state() -> Dict[str, Any]:
    """Capture Python/NumPy/PyTorch RNG states for exact resume."""
    state: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG states captured by :func:`capture_rng_state`."""
    if not isinstance(state, dict):
        return

    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "numpy_random_state" in state:
        np.random.set_state(state["numpy_random_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])

    if torch.cuda.is_available() and "torch_cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])
