"""Technical indicator computation.

All indicators use vectorized pandas/numpy operations and never access future bars.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, min_periods=window, adjust=False).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series: pd.Series, window: int = 20, sigma: float = 2.0):
    mid = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + sigma * std
    lower = mid - sigma * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return mid, upper, lower, width


def compute_rolling_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    log_ret = np.log((series / series.shift(1)).astype(float))
    return log_ret.rolling(window=window, min_periods=window).std()


def compute_log_return(series: pd.Series, horizon: int = 1) -> pd.Series:
    return np.log((series / series.shift(horizon)).astype(float))


def add_technical_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add technical indicator columns to *df* based on feature config."""
    tech_cfg = config.get("features", {}).get("technical", {})
    indicators = tech_cfg.get("enabled_indicators", [])
    close = df["close"]

    if "sma" in indicators:
        for w in tech_cfg.get("moving_average_windows", [10, 20, 50]):
            df[f"tech_sma_{w}"] = compute_sma(close, w)

    if "ema" in indicators:
        for w in tech_cfg.get("moving_average_windows", [10, 20, 50]):
            df[f"tech_ema_{w}"] = compute_ema(close, w)

    if "rsi" in indicators:
        rsi_w = tech_cfg.get("rsi_window", 14)
        df[f"tech_rsi_{rsi_w}"] = compute_rsi(close, rsi_w)

    if "macd" in indicators:
        fast = tech_cfg.get("macd_fast", 12)
        slow = tech_cfg.get("macd_slow", 26)
        sig = tech_cfg.get("macd_signal", 9)
        macd_line, sig_line, hist = compute_macd(close, fast, slow, sig)
        df["tech_macd"] = macd_line
        df["tech_macd_signal"] = sig_line
        df["tech_macd_hist"] = hist

    if "bollinger" in indicators:
        bw = tech_cfg.get("bollinger_window", 20)
        bs = tech_cfg.get("bollinger_sigma", 2.0)
        mid, upper, lower, width = compute_bollinger(close, bw, bs)
        df["tech_boll_mid"] = mid
        df["tech_boll_upper"] = upper
        df["tech_boll_lower"] = lower
        df["tech_boll_width"] = width

    if "rolling_volatility" in indicators:
        rv_w = tech_cfg.get("rolling_volatility_window", 20)
        df[f"tech_roll_vol_{rv_w}"] = compute_rolling_volatility(close, rv_w)

    if "log_return" in indicators:
        lr_h = tech_cfg.get("log_return_horizon", 1)
        df[f"tech_logret_{lr_h}"] = compute_log_return(close, lr_h)

    return df
