"""Evaluation metrics — cumulative return, Sharpe, Sortino, drawdown, etc."""

import numpy as np
from typing import Any, Dict, List, Optional


def compute_metrics(
    equity_curve: np.ndarray,
    trade_log: Optional[List[Dict[str, Any]]] = None,
    periods_per_year: int = 6048,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute all required evaluation metrics from an equity curve.

    Parameters
    ----------
    equity_curve : np.ndarray
        1-D array of equity values (length >= 2).
    trade_log : list of dict, optional
        Per-trade records with keys ``pnl``, ``pyramid_steps``, ``martingale_steps``, ``forced_liquidation``.
    periods_per_year : int
        Annualization factor (default 6048 for hourly weekday-style).
    risk_free_rate : float
        Per-period risk-free rate.

    Returns
    -------
    dict
        Metric name → value.
    """
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    if equity_curve.size == 0:
        equity_curve = np.asarray([1.0], dtype=np.float64)
    equity_curve = np.nan_to_num(equity_curve, nan=0.0, posinf=0.0, neginf=0.0)
    N = len(equity_curve)
    eps = 1e-12

    initial_equity = equity_curve[0] if N > 0 else 1.0
    final_equity = equity_curve[-1] if N > 0 else 1.0

    # Step returns
    if N >= 2:
        step_returns = np.diff(equity_curve) / np.maximum(np.abs(equity_curve[:-1]), eps)
        step_returns = np.nan_to_num(step_returns, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        step_returns = np.array([0.0])

    # Cumulative return
    cumulative_return = final_equity / max(initial_equity, eps) - 1.0

    # Annualized return
    n_periods = max(N - 1, 1)
    if final_equity > 0 and initial_equity > 0:
        ann_return = (final_equity / initial_equity) ** (periods_per_year / n_periods) - 1.0
    else:
        ann_return = -1.0

    # Annualized volatility
    vol = float(np.std(step_returns))
    ann_vol = vol * np.sqrt(periods_per_year)

    # Sharpe ratio
    excess = step_returns - risk_free_rate
    sharpe = float(np.mean(excess)) / max(float(np.std(step_returns)), eps) * np.sqrt(periods_per_year)

    # Sortino ratio
    neg_returns = step_returns[step_returns < risk_free_rate] - risk_free_rate
    downside_std = float(np.std(neg_returns)) if len(neg_returns) > 0 else eps
    sortino = float(np.mean(excess)) / max(downside_std, eps) * np.sqrt(periods_per_year)

    # Maximum drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / np.maximum(peak, eps)
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Trade-based metrics
    win_rate = 0.0
    total_trades = 0
    avg_pyramid_steps = 0.0
    avg_martingale_steps = 0.0
    liquidation_count = 0
    turnover = 0.0

    if trade_log:
        valid_trades = []
        for t in trade_log:
            if not isinstance(t, dict):
                continue
            pnl = float(np.nan_to_num(t.get("pnl", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
            pyramid_steps = int(t.get("pyramid_steps", 0) or 0)
            martingale_steps = int(t.get("martingale_steps", 0) or 0)
            forced_liquidation = bool(t.get("forced_liquidation", False))
            notional = float(np.nan_to_num(t.get("notional", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
            valid_trades.append({
                "pnl": pnl,
                "pyramid_steps": max(pyramid_steps, 0),
                "martingale_steps": max(martingale_steps, 0),
                "forced_liquidation": forced_liquidation,
                "notional": abs(notional),
            })

        total_trades = len(valid_trades)
        wins = sum(1 for t in valid_trades if t.get("pnl", 0) > 0)
        win_rate = wins / max(total_trades, 1)
        avg_pyramid_steps = float(np.mean([t.get("pyramid_steps", 0) for t in valid_trades])) if total_trades > 0 else 0.0
        avg_martingale_steps = float(np.mean([t.get("martingale_steps", 0) for t in valid_trades])) if total_trades > 0 else 0.0
        liquidation_count = sum(1 for t in valid_trades if t.get("forced_liquidation", False))
        notionals = [abs(t.get("notional", 0)) for t in valid_trades]
        avg_eq = float(np.mean(equity_curve)) if N > 0 else 1.0
        turnover = sum(notionals) / max(avg_eq, eps)

    liquidation_rate = liquidation_count / max(total_trades, 1) if total_trades > 0 else 0.0

    def _finite(v: float) -> float:
        return float(v) if np.isfinite(v) else 0.0

    return {
        "cumulative_return": _finite(cumulative_return),
        "annualized_return": _finite(ann_return),
        "annualized_volatility": _finite(ann_vol),
        "max_drawdown": _finite(max_drawdown),
        "sharpe_ratio": _finite(sharpe),
        "sortino_ratio": _finite(sortino),
        "win_rate": _finite(win_rate),
        "total_trades": total_trades,
        "turnover": _finite(turnover),
        "avg_pyramid_steps": _finite(avg_pyramid_steps),
        "avg_martingale_steps": _finite(avg_martingale_steps),
        "liquidation_count": liquidation_count,
        "liquidation_rate": _finite(liquidation_rate),
    }
