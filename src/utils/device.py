"""Device selection utility for PyTorch."""

import torch
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """Return a ``torch.device`` based on *preference*.

    Parameters
    ----------
    preference : str
        One of ``"auto"``, ``"cpu"``, ``"cuda"`` (aliases like ``"gpu"`` and
        explicit CUDA device strings such as ``"cuda:0"`` are also supported).
        ``"auto"`` picks CUDA when available, else CPU.
    """
    pref = str(preference).strip().lower()

    if pref in {"gpu", "cuda"} and torch.cuda.is_available():
        device = torch.device("cuda")
    elif pref.startswith("cuda:") and torch.cuda.is_available():
        device = torch.device(pref)
    elif pref == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Selected device: %s", device)
    return device
