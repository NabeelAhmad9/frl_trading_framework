"""Microstructure proxy features derived from OHLCV data.

These are proxy measures — not true bid/ask or depth measurements.
"""

from typing import Dict

import numpy as np
import pandas as pd


def compute_spread_proxy_hl(df: pd.DataFrame) -> pd.Series:
    """High-low range based spread proxy."""
    return (df["high"] - df["low"]) / df["close"].replace(0, np.nan)


def compute_spread_proxy_co(df: pd.DataFrame) -> pd.Series:
    """Close-open range based spread proxy."""
    return np.abs(df["close"] - df["open"]) / df["close"].replace(0, np.nan)


def compute_price_change_rate(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Rolling percentage price change rate."""
    return df["close"].pct_change(periods=window)


def compute_realized_volatility_proxy(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling standard deviation of log returns as volatility proxy."""
    log_ret = np.log((df["close"] / df["close"].shift(1)).astype(float))
    return log_ret.rolling(window=window, min_periods=window).std()


def compute_session_label(df: pd.DataFrame, tz: str = "UTC") -> pd.Series:
    """Label trading sessions based on hour of day.
    
    Warning: Sessions are indicative proxies based on UTC house hours.
    """
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        timestamps = pd.to_datetime(df["timestamp"])
    else:
        timestamps = df["timestamp"]
        
    hours = timestamps.dt.hour
    labels = pd.Series("off_hours", index=df.index)
    
    # Asia: 00:00 - 08:00 UTC
    labels[(hours >= 0) & (hours < 8)] = "asia"
    # London: 08:00 - 13:00 UTC
    labels[(hours >= 8) & (hours < 13)] = "london"
    # Overlap / Early NY: 13:00 - 17:00 UTC
    labels[(hours >= 13) & (hours < 17)] = "new_york"
    # Late NY: 17:00 - 22:00 UTC
    labels[(hours >= 17) & (hours < 22)] = "new_york"
    # Off-hours: 22:00 - 00:00 UTC
    return labels


def add_microstructure_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add microstructure proxy columns to *df* based on feature config."""
    micro_cfg = config.get("features", {}).get("microstructure", {})

    method = micro_cfg.get("spread_proxy_method", "hl_ratio")
    if method == "hl_ratio":
        df["micro_spread_proxy"] = compute_spread_proxy_hl(df)
    elif method == "co_ratio":
        df["micro_spread_proxy"] = compute_spread_proxy_co(df)
    else:
        df["micro_spread_proxy"] = compute_spread_proxy_hl(df)

    pcr_w = micro_cfg.get("price_change_rate_window", 5)
    df[f"micro_price_change_rate_{pcr_w}"] = compute_price_change_rate(df, pcr_w)

    rv_w = micro_cfg.get("realized_volatility_window", 20)
    df[f"micro_realized_vol_{rv_w}"] = compute_realized_volatility_proxy(df, rv_w)

    if micro_cfg.get("session_proxy_enabled", False):
        df["micro_session_label"] = compute_session_label(df)

    return df
