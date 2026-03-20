"""Centralized logging setup for FRL Trading Framework."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


_CONFIGURED = False


def setup_logger(
    name: str = "frl",
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
) -> logging.Logger:
    """Create or retrieve a configured logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : str
        Logging level string (DEBUG, INFO, WARNING, ERROR).
    log_file : path, optional
        If provided, add a file handler writing to this path.
    console : bool
        Whether to add a stderr handler.
    """
    global _CONFIGURED
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not _CONFIGURED:
        if console:
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(fmt)
            logger.addHandler(sh)
        _CONFIGURED = True

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "frl") -> logging.Logger:
    """Retrieve an existing logger by name, creating a basic one if needed."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


def write_log_file(
    log_file: Optional[Union[str, Path]],
    lines,
    mode: str = "a",
) -> Optional[Path]:
    """Write one or more plain-text lines to a canonical workflow log file."""
    if log_file is None:
        return None

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(lines, str):
        payload = [lines]
    else:
        payload = [str(line) for line in lines]

    with open(log_path, mode, encoding="utf-8") as fh:
        for line in payload:
            fh.write(f"{str(line).rstrip()}\n")

    return log_path
