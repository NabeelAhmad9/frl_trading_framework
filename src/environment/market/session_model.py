"""Session model — map timestamps to trading session labels and multipliers."""

from typing import Any, Dict, Tuple
import pandas as pd


# Session boundaries in UTC
_SESSION_BOUNDARIES = [
    ("Asia",     0,  8),
    ("London",   8, 13),
    ("Overlap", 13, 17),
    ("NewYork", 17, 22),
    ("OffHours", 22, 24),
]


def label_session(hour_utc: int) -> str:
    """Return session name for a given UTC hour."""
    for name, start, end in _SESSION_BOUNDARIES:
        if start <= hour_utc < end:
            return name
    return "OffHours"


def session_multiplier(session: str, config: Dict[str, Any]) -> float:
    """Return session-specific slippage/liquidity multiplier."""
    multipliers = {
        "Asia": 1.2,
        "London": 0.8,
        "Overlap": 0.7,
        "NewYork": 0.9,
        "OffHours": 1.5,
    }
    return multipliers.get(session, 1.0)


def add_session_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add session_label column to a DataFrame with a datetime index or 'timestamp' column."""
    if "timestamp" in df.columns:
        hours = pd.to_datetime(df["timestamp"]).dt.hour
    elif isinstance(df.index, pd.DatetimeIndex):
        hours = df.index.hour
    else:
        df["session_label"] = "unknown"
        return df
    df["session_label"] = hours.map(label_session)
    return df
