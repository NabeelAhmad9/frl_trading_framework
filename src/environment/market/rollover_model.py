"""Rollover model — overnight financing approximation."""

from typing import Any, Dict
import pandas as pd


def compute_rollover_cost(
    timestamp: object,
    direction: int,
    total_lots: float,
    config: Dict[str, Any],
) -> float:
    """Compute rollover cost in account currency.

    Parameters
    ----------
    timestamp : pd.Timestamp
        Bar timestamp (UTC).
    direction : int
        -1=short, 0=flat, 1=long.
    total_lots : float
        Open lots.
    config : dict
        Rollover config block.

    Returns
    -------
    float
        Rollover cost (positive = cost to agent, negative = credit).
    """
    if direction == 0 or total_lots == 0:
        return 0.0

    enabled = config.get("enabled", True)
    if not enabled:
        return 0.0

    cutoff_hour = config.get("cutoff_hour_utc", 22)
    triple_day = config.get("triple_rollover_day", "Wednesday")
    annual_long = config.get("annual_rate_long", 0.02)
    annual_short = config.get("annual_rate_short", -0.015)

    ts = pd.Timestamp(timestamp)
    if ts.hour != cutoff_hour:
        return 0.0

    # Daily rate
    if direction == 1:
        daily_rate = annual_long / 365.0
    else:
        daily_rate = annual_short / 365.0

    # Triple rollover on configured day
    multiplier = 1
    if ts.day_name() == triple_day:
        multiplier = 3

    lot_units = 100000
    cost = daily_rate * total_lots * lot_units * multiplier
    return cost
