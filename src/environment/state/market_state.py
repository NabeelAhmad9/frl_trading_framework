"""Market state container — immutable snapshot of market data at one timestep."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MarketState:
    """Immutable market state at a single bar."""
    timestamp: object  # pd.Timestamp
    pair: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread_proxy: float = 0.0
    volatility_proxy: float = 0.0
    session_label: str = "unknown"
    rollover_flag: bool = False
    pip_size: float = 0.0001
    price_precision: int = 5
