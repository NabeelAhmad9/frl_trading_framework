"""Market simulator — sequential market progression with state snapshots."""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.environment.state.market_state import MarketState
from src.environment.market.session_model import label_session


PAIR_METADATA = {
    "EURUSD": {"pip_size": 0.0001, "price_precision": 5, "quote_currency": "USD"},
    "GBPUSD": {"pip_size": 0.0001, "price_precision": 5, "quote_currency": "USD"},
    "USDJPY": {"pip_size": 0.01, "price_precision": 3, "quote_currency": "JPY"},
    "AUDUSD": {"pip_size": 0.0001, "price_precision": 5, "quote_currency": "USD"},
}


class MarketSimulator:
    """Iterate through dataset rows emitting MarketState snapshots."""

    def __init__(self, df: pd.DataFrame, pair: str, config: Dict[str, Any]):
        self.df = df.reset_index(drop=True)
        self.pair = pair
        self.config = config
        meta = PAIR_METADATA.get(pair, {"pip_size": 0.0001, "price_precision": 5, "quote_currency": "USD"})
        self.pip_size = meta["pip_size"]
        self.price_precision = meta["price_precision"]
        self.quote_currency = meta["quote_currency"]
        self._cursor = 0
        self._length = len(df)

        # Precompute market arrays once to avoid per-step pandas indexing overhead.
        def _series_or_default(primary: str, secondary: Optional[str], default: float = 0.0) -> np.ndarray:
            if primary in self.df.columns:
                return self.df[primary].to_numpy(dtype=np.float64, copy=False)
            if secondary and secondary in self.df.columns:
                return self.df[secondary].to_numpy(dtype=np.float64, copy=False)
            return np.full(self._length, float(default), dtype=np.float64)

        if "timestamp" in self.df.columns:
            ts_series = self.df["timestamp"]
        else:
            ts_series = pd.Series([None] * self._length)

        self._timestamps = ts_series.to_numpy(copy=False)
        self._open = _series_or_default("open", "Open")
        self._high = _series_or_default("high", "High")
        self._low = _series_or_default("low", "Low")
        self._close = _series_or_default("close", "Close")
        self._volume = _series_or_default("volume", "Volume")
        self._spread_proxy = _series_or_default("spread_proxy_hl", "spread_proxy")
        self._volatility_proxy = _series_or_default("realized_volatility_proxy", "volatility_proxy")

        rollover_cfg = self.config.get("environment", self.config).get("rollover", {})
        cutoff = int(rollover_cfg.get("cutoff_hour_utc", 22))

        if self._length > 0:
            ts_dt = pd.to_datetime(ts_series, errors="coerce")
            hours = ts_dt.dt.hour.fillna(0).astype(np.int16).to_numpy()
        else:
            hours = np.zeros((0,), dtype=np.int16)

        self._session_labels = np.array([label_session(int(h)) for h in hours], dtype=object)
        self._rollover_flags = (hours == cutoff)

        # Tiny LRU cache: avoids rebuilding the same state objects repeatedly within a step.
        self._state_cache: "OrderedDict[int, MarketState]" = OrderedDict()

    @property
    def pair_metadata(self) -> Dict[str, Any]:
        return {
            "pip_size": self.pip_size,
            "price_precision": self.price_precision,
            "quote_currency": self.quote_currency,
        }

    def reset(self, start_index: int = 0) -> None:
        self._cursor = start_index

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def length(self) -> int:
        return self._length

    def has_next(self) -> bool:
        return self._cursor < self._length - 1

    def current_state(self) -> MarketState:
        return self._row_to_state(self._cursor)

    def next_state(self) -> Optional[MarketState]:
        if self._cursor + 1 < self._length:
            return self._row_to_state(self._cursor + 1)
        return None

    def advance(self) -> MarketState:
        self._cursor += 1
        return self.current_state()

    def _row_to_state(self, idx: int) -> MarketState:
        if idx in self._state_cache:
            state = self._state_cache.pop(idx)
            self._state_cache[idx] = state
            return state

        state = MarketState(
            timestamp=self._timestamps[idx],
            pair=self.pair,
            open=float(self._open[idx]),
            high=float(self._high[idx]),
            low=float(self._low[idx]),
            close=float(self._close[idx]),
            volume=float(self._volume[idx]),
            spread_proxy=float(self._spread_proxy[idx]),
            volatility_proxy=float(self._volatility_proxy[idx]),
            session_label=str(self._session_labels[idx]),
            rollover_flag=bool(self._rollover_flags[idx]),
            pip_size=self.pip_size,
            price_precision=self.price_precision,
        )

        self._state_cache[idx] = state
        if len(self._state_cache) > 8:
            self._state_cache.popitem(last=False)
        return state

    def get_market_window(self, end_idx: int, window_length: int) -> List[MarketState]:
        """Get a window of market states ending at end_idx."""
        start = max(0, end_idx - window_length + 1)
        return [self._row_to_state(i) for i in range(start, end_idx + 1)]
