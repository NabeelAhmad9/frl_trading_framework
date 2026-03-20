"""Performance report — produce summary tables from metrics."""

import csv
import numpy as np
from pathlib import Path
from typing import Any, Dict, Union, Iterable

from src.utils.logger import get_logger

logger = get_logger(__name__)


DEFAULT_SCALAR_METRICS = (
    "cumulative_return",
    "sharpe_ratio",
    "max_drawdown",
    "turnover",
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def save_performance_summary(metrics: Dict[str, float], output_path: Path) -> Path:
    """Write performance_summary.csv with one row per metric."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for name, val in sorted(metrics.items()):
            writer.writerow([name, _finite_float(val)])

    logger.info("Performance summary saved → %s", output_path)
    return output_path


def save_risk_metrics(metrics: Dict[str, float], output_path: Path) -> Path:
    """Write risk_metrics.csv with risk-related subset."""
    risk_keys = [
        "max_drawdown", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
        "liquidation_count", "liquidation_rate",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in risk_keys:
            if key in metrics:
                writer.writerow([key, _finite_float(metrics[key])])

    logger.info("Risk metrics saved → %s", output_path)
    return output_path


def save_equity_curve(equity_curve: Union[np.ndarray, Iterable[float]], output_path: Path) -> Path:
    """Save raw equity curve time-series to CSV.

    Parameters
    ----------
    equity_curve : array-like
        The sequence of equity values.
    output_path : Path
        Path to save equity_curve.csv.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.size > 0:
        equity = np.nan_to_num(equity, nan=0.0, posinf=0.0, neginf=0.0)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "equity_value"])
        for i, val in enumerate(equity):
            writer.writerow([i, _finite_float(val)])

    logger.info("Equity curve CSV saved → %s", output_path)
    return output_path


def save_drawdown_curve(equity_curve: Union[np.ndarray, Iterable[float]], output_path: Path) -> Path:
    """Save raw drawdown curve time-series to CSV.

    Parameters
    ----------
    equity_curve : array-like
        The sequence of equity values from which drawdown is derived.
    output_path : Path
        Path to save drawdown_curve.csv.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.size > 0:
        eq = np.nan_to_num(eq, nan=0.0, posinf=0.0, neginf=0.0)
    if len(eq) == 0:
        dd = np.array([])
    else:
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.maximum(peak, 1e-12)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "drawdown_value"])
        for i, val in enumerate(dd):
            writer.writerow([i, _finite_float(val)])

    logger.info("Drawdown curve CSV saved → %s", output_path)
    return output_path


def save_comparison_table(
    agent_metrics: Dict[str, Dict[str, float]],
    output_path: Path,
) -> Path:
    """Write a side-by-side comparison CSV.

    Parameters
    ----------
    agent_metrics : dict
        Mapping agent_name → metrics dict.
    output_path : Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agents = sorted(agent_metrics.keys())
    all_metrics = set()
    for m in agent_metrics.values():
        all_metrics.update(m.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + agents)
        for metric in sorted(all_metrics):
            row = [metric] + [
                _finite_float(agent_metrics[a][metric]) if metric in agent_metrics[a] else ""
                for a in agents
            ]
            writer.writerow(row)

    logger.info("Comparison table saved → %s", output_path)
    return output_path


def save_metric_snapshots(
    metrics: Dict[str, float],
    output_dir: Path,
    metric_names=DEFAULT_SCALAR_METRICS,
) -> Dict[str, Path]:
    """Save canonical one-metric-per-file CSV snapshots under *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for metric_name in metric_names:
        if metric_name not in metrics:
            continue

        output_path = output_dir / f"{metric_name}.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow([metric_name, _finite_float(metrics[metric_name])])

        written[metric_name] = output_path

    return written


def save_actions_sequence(actions, output_path: Path) -> Path:
    """Persist a canonical actions sequence CSV for evaluation-style runs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "global_step",
        "step",
        "timestamp",
        "action_id",
        "action_name",
        "was_legal",
        "executed_action_name",
        "direction_before",
        "direction_after",
        "total_lots_after",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for action in actions:
            row = {name: action.get(name, "") for name in fieldnames}
            writer.writerow(row)

    logger.info("Actions sequence saved → %s", output_path)
    return output_path


def save_statistical_tests(results: Dict[str, Any], output_path: Path) -> Path:
    """Write statistical comparison results to a stable CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["comparison", "method", "statistic", "p_value", "significant", "reason"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if "result" in results and results.get("result") == "not_applicable":
            writer.writerow({
                "comparison": "overall",
                "method": "not_applicable",
                "statistic": "",
                "p_value": "",
                "significant": "",
                "reason": results.get("reason", "not_applicable"),
            })
        else:
            for comparison_name, result in results.items():
                writer.writerow({
                    "comparison": comparison_name,
                    "method": result.get("method", "not_applicable"),
                    "statistic": result.get("statistic", ""),
                    "p_value": result.get("p_value", ""),
                    "significant": result.get("significant", ""),
                    "reason": result.get("reason", ""),
                })

    logger.info("Statistical tests saved → %s", output_path)
    return output_path


def save_trade_log(trade_log, output_path: Path) -> Path:
    """Persist a canonical trade log CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "step",
        "pnl",
        "direction_before",
        "direction",
        "forced_liquidation",
        "pyramid_steps",
        "martingale_steps",
        "notional",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for event in trade_log or []:
            writer.writerow({
                "step": int(event.get("step", 0)),
                "pnl": _finite_float(event.get("pnl", 0.0)),
                "direction_before": int(event.get("direction_before", 0)),
                "direction": int(event.get("direction", 0)),
                "forced_liquidation": bool(event.get("forced_liquidation", False)),
                "pyramid_steps": int(event.get("pyramid_steps", 0)),
                "martingale_steps": int(event.get("martingale_steps", 0)),
                "notional": _finite_float(event.get("notional", 0.0)),
            })

    logger.info("Trade log saved → %s", output_path)
    return output_path
