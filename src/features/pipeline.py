"""Feature pipeline — orchestrate creation, warm-up, scaling, and final schema."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.features.technical import add_technical_features
from src.features.microstructure import add_microstructure_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Canonical column ordering groups
_MARKET_IDENTITY = ["timestamp", "open", "high", "low", "close", "volume"]


def _get_warmup_requirement(config: Dict) -> int:
    """Determine the maximum lookback needed by enabled features."""
    tech_cfg = config.get("features", {}).get("technical", {})
    micro_cfg = config.get("features", {}).get("microstructure", {})
    windows = []

    # Technical windows
    windows.extend(tech_cfg.get("moving_average_windows", [50]))
    windows.append(tech_cfg.get("rsi_window", 14))
    windows.append(tech_cfg.get("macd_slow", 26) + tech_cfg.get("macd_signal", 9))
    windows.append(tech_cfg.get("bollinger_window", 20))
    windows.append(tech_cfg.get("rolling_volatility_window", 20))

    # Microstructure windows
    windows.append(micro_cfg.get("realized_volatility_window", 20))
    windows.append(micro_cfg.get("price_change_rate_window", 5))

    return max(windows) if windows else 50


def _get_feature_columns(df: pd.DataFrame, retained_raw: List[str]) -> List[str]:
    """Return the list of numeric feature columns excluding retained raw columns."""
    exclude = set(retained_raw) | {"micro_session_label"}
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]


def _enforce_column_order(df: pd.DataFrame, retained_raw: List[str]) -> pd.DataFrame:
    """Enforce canonical column order: market -> retained -> technical -> microstructure."""
    market_cols = [c for c in _MARKET_IDENTITY if c in df.columns]
    retained_extra = [c for c in retained_raw if c not in market_cols and c in df.columns]
    tech_cols = sorted([c for c in df.columns if c.startswith("tech_")])
    micro_cols = sorted([c for c in df.columns if c.startswith("micro_")])
    other_cols = [c for c in df.columns if c not in market_cols + retained_extra + tech_cols + micro_cols]
    ordered = market_cols + retained_extra + tech_cols + micro_cols + other_cols
    return df[ordered]


def run_feature_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full feature engineering pipeline.

    1. Compute features on train and test separately.
    2. Compute warm-up window. Trim NaN rows.
    3. Fit scaler on train only, transform both.
    4. Enforce column order.
    """
    pipeline_cfg = config.get("features", {}).get("pipeline", {})
    groups = pipeline_cfg.get("selected_feature_groups", ["technical", "microstructure"])
    retained_raw = pipeline_cfg.get("retained_raw_columns", list(_MARKET_IDENTITY))
    norm_cfg = pipeline_cfg.get("normalization", {})

    # --- Compute features on combined context for test warm-up ---
    warmup = _get_warmup_requirement(config)
    logger.info("Feature warm-up requirement: %d bars", warmup)

    # For test: borrow tail of train for warm-up context
    # This ensures features at the beginning of test set have full lookback history.
    train_tail = train_df.tail(warmup).copy()
    test_with_context = pd.concat([train_tail, test_df], ignore_index=True)

    # Compute features on training data
    if "technical" in groups:
        train_df = add_technical_features(train_df, config)
    if "microstructure" in groups:
        train_df = add_microstructure_features(train_df, config)

    # Compute features on test data (with warm-up context)
    if "technical" in groups:
        test_with_context = add_technical_features(test_with_context, config)
    if "microstructure" in groups:
        test_with_context = add_microstructure_features(test_with_context, config)

    # Strip the borrowed warm-up rows from test
    test_featured = test_with_context.iloc[len(train_tail):].reset_index(drop=True)

    # Trim NaN rows from warm-up
    train_df = train_df.dropna().reset_index(drop=True)
    test_featured = test_featured.dropna().reset_index(drop=True)

    logger.info("After warm-up trim: train=%d, test=%d", len(train_df), len(test_featured))

    # --- Scaling (train-only fit) ---
    if norm_cfg.get("enabled", False):
        feature_cols = _get_feature_columns(train_df, retained_raw)
        scaler_type = norm_cfg.get("scaler_type", "standard")
        clip_min = norm_cfg.get("clip_min", -5.0)
        clip_max = norm_cfg.get("clip_max", 5.0)

        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = None

        if scaler is not None and feature_cols:
            scaler.fit(train_df[feature_cols])
            train_df[feature_cols] = scaler.transform(train_df[feature_cols])
            test_featured[feature_cols] = scaler.transform(test_featured[feature_cols])

            # Clip
            train_df[feature_cols] = train_df[feature_cols].clip(clip_min, clip_max)
            test_featured[feature_cols] = test_featured[feature_cols].clip(clip_min, clip_max)

            logger.info("Applied %s scaling and clipping [%.1f, %.1f]", scaler_type, clip_min, clip_max)

    # --- Enforce column order ---
    train_df = _enforce_column_order(train_df, retained_raw)
    test_featured = _enforce_column_order(test_featured, retained_raw)

    # Verify schema match
    if list(train_df.columns) != list(test_featured.columns):
        raise ValueError("Train and test feature schemas do not match after pipeline!")

    # Verify no remaining NaNs in feature columns
    feat_cols = _get_feature_columns(train_df, retained_raw)
    for label, df_check in [("train", train_df), ("test", test_featured)]:
        nan_count = df_check[feat_cols].isna().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Residual NaNs in {label} feature columns: {nan_count}")

    return train_df, test_featured
