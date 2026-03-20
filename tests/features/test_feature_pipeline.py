"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features.technical import add_technical_features, compute_sma, compute_rsi
from src.features.microstructure import add_microstructure_features
from src.features.pipeline import run_feature_pipeline


def _make_df(n=300):
    np.random.seed(42)
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    close = 1.1 + np.cumsum(np.random.randn(n) * 0.0005)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close + np.random.randn(n) * 0.0001,
        "high": close + np.abs(np.random.randn(n)) * 0.001,
        "low": close - np.abs(np.random.randn(n)) * 0.001,
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })


def _base_config():
    return {
        "features": {
            "technical": {
                "enabled_indicators": ["sma", "ema", "rsi", "macd", "bollinger", "rolling_volatility", "log_return"],
                "moving_average_windows": [10, 20],
                "rsi_window": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bollinger_window": 20,
                "bollinger_sigma": 2.0,
                "rolling_volatility_window": 20,
                "log_return_horizon": 1,
            },
            "microstructure": {
                "spread_proxy_method": "hl_ratio",
                "realized_volatility_window": 20,
                "price_change_rate_window": 5,
                "session_proxy_enabled": True,
            },
            "pipeline": {
                "selected_feature_groups": ["technical", "microstructure"],
                "normalization": {
                    "enabled": True,
                    "scaler_type": "standard",
                    "clip_min": -5.0,
                    "clip_max": 5.0,
                },
                "retained_raw_columns": ["timestamp", "open", "high", "low", "close", "volume"],
                "output_column_order_policy": "canonical",
            },
        }
    }


class TestTechnical:
    def test_sma(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = compute_sma(s, 3)
        assert np.isnan(result.iloc[0])
        assert result.iloc[2] == pytest.approx(2.0)

    def test_rsi_range(self):
        np.random.seed(42)
        s = pd.Series(1.0 + np.cumsum(np.random.randn(100) * 0.01))
        rsi = compute_rsi(s, 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


class TestMicrostructure:
    def test_spread_proxy(self):
        df = _make_df(50)
        result = add_microstructure_features(df.copy(), _base_config())
        assert "micro_spread_proxy" in result.columns
        assert (result["micro_spread_proxy"].dropna() >= 0).all()


class TestPipeline:
    def test_full_pipeline(self):
        df = _make_df(300)
        # 80/20 split
        split_idx = 240
        train = df.iloc[:split_idx].reset_index(drop=True)
        test = df.iloc[split_idx:].reset_index(drop=True)
        config = _base_config()
        train_out, test_out = run_feature_pipeline(train, test, config)
        # Schema match
        assert list(train_out.columns) == list(test_out.columns)
        # No NaNs in features
        feat_cols = [c for c in train_out.columns if c.startswith("tech_") or c.startswith("micro_")]
        feat_cols = [c for c in feat_cols if c != "micro_session_label"]
        assert train_out[feat_cols].isna().sum().sum() == 0
        assert test_out[feat_cols].isna().sum().sum() == 0

    def test_train_only_scaling(self):
        df = _make_df(300)
        split_idx = 240
        train = df.iloc[:split_idx].reset_index(drop=True)
        test = df.iloc[split_idx:].reset_index(drop=True)
        config = _base_config()
        train_out, test_out = run_feature_pipeline(train, test, config)
        # After standard scaling, train features should have ~0 mean
        feat_cols = [c for c in train_out.columns if c.startswith("tech_") and train_out[c].dtype == np.float64]
        if feat_cols:
            train_means = train_out[feat_cols].mean().abs()
            assert (train_means < 0.1).all()  # approximately zero-centered

    def test_no_scaling(self):
        df = _make_df(300)
        split_idx = 240
        train = df.iloc[:split_idx].reset_index(drop=True)
        test = df.iloc[split_idx:].reset_index(drop=True)
        config = _base_config()
        config["features"]["pipeline"]["normalization"]["enabled"] = False
        train_out, test_out = run_feature_pipeline(train, test, config)
        assert list(train_out.columns) == list(test_out.columns)

    def test_column_ordering(self):
        df = _make_df(300)
        split_idx = 240
        train = df.iloc[:split_idx].reset_index(drop=True)
        test = df.iloc[split_idx:].reset_index(drop=True)
        config = _base_config()
        train_out, _ = run_feature_pipeline(train, test, config)
        cols = list(train_out.columns)
        # Market columns should come first
        assert cols[0] == "timestamp"
        assert "close" in cols[:6]
