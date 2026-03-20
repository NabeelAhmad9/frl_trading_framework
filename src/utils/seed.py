"""Seed management for reproducibility."""

import os
import random

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 42, deterministic_torch: bool = False) -> None:
    """Set seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value.
    deterministic_torch : bool
        If True, enable PyTorch deterministic mode (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Global seed set to %d (deterministic_torch=%s)", seed, deterministic_torch)
