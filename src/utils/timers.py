"""Timing utilities for profiling long-running stages."""

import time
from contextlib import contextmanager

from src.utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def timer(label: str = "block"):
    """Context manager that logs elapsed wall-clock time for *label*."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info("Timer [%s]: %.3f seconds", label, elapsed)


class StageTimer:
    """Accumulates timings for named stages."""

    def __init__(self):
        self._timings: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        elapsed = time.perf_counter() - self._starts.pop(name)
        self._timings[name] = self._timings.get(name, 0.0) + elapsed
        return elapsed

    @property
    def timings(self) -> dict[str, float]:
        return dict(self._timings)
