"""Trade-log helpers used by trainer and evaluation pipelines."""

from typing import Any, Dict, Optional


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def build_trade_event(info: Dict[str, Any], step: int) -> Optional[Dict[str, Any]]:
    """Return a canonical trade event dict or ``None`` for non-trade steps.

    A step is treated as a trade event when any of the following happens:
    - direction changes,
    - realized PnL is non-zero,
    - forced liquidation occurs.
    """
    direction_before = _to_int(info.get("direction_before", info.get("direction", 0)), 0)
    direction_after = _to_int(info.get("direction", direction_before), direction_before)
    forced_liquidation = bool(info.get("forced_liquidation", False))
    realized_pnl = _to_float(info.get("realized_pnl", 0.0), 0.0)

    is_trade_event = (
        forced_liquidation
        or direction_before != direction_after
        or abs(realized_pnl) > 1e-12
    )
    if not is_trade_event:
        return None

    pyramid_before = _to_int(info.get("pyramid_levels_before", info.get("pyramid_levels", 0)), 0)
    pyramid_after = _to_int(info.get("pyramid_levels", pyramid_before), pyramid_before)
    martingale_before = _to_int(info.get("martingale_steps_before", info.get("martingale_steps", 0)), 0)
    martingale_after = _to_int(info.get("martingale_steps", martingale_before), martingale_before)

    lots_before = abs(_to_float(info.get("total_lots_before", info.get("total_lots", 0.0)), 0.0))
    lots_after = abs(_to_float(info.get("total_lots_after", info.get("total_lots", 0.0)), 0.0))
    turnover_delta = abs(_to_float(info.get("turnover_delta", 0.0), 0.0))

    if turnover_delta > 0.0:
        notional = turnover_delta
    else:
        equity_ref = abs(_to_float(info.get("equity_after", info.get("equity", 0.0)), 0.0))
        notional = max(lots_before, lots_after) * equity_ref

    return {
        "step": int(step),
        "pnl": float(realized_pnl),
        "direction_before": int(direction_before),
        "direction": int(direction_after),
        "forced_liquidation": bool(forced_liquidation),
        "pyramid_steps": int(max(pyramid_before, pyramid_after)),
        "martingale_steps": int(max(martingale_before, martingale_after)),
        "notional": float(max(notional, 0.0)),
    }
