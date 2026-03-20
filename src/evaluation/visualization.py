"""Visualization — equity curves, drawdown, comparisons."""

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_equity_curve(
    equity_curve: np.ndarray,
    output_path: Path,
    title: str = "Equity Curve",
    fmt: str = "pdf",
) -> Optional[Path]:
    """Save equity curve plot."""
    if not HAS_MPL:
        logger.warning("matplotlib not available; skipping equity curve plot")
        return None
    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.size == 0:
        logger.warning("Empty equity curve; skipping equity curve plot")
        return None
    eq = np.nan_to_num(eq, nan=0.0, posinf=0.0, neginf=0.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=150)
    plt.close(fig)
    logger.info("Equity curve saved → %s", output_path)
    return output_path


def plot_drawdown_curve(
    equity_curve: np.ndarray,
    output_path: Path,
    title: str = "Drawdown Curve",
    fmt: str = "pdf",
) -> Optional[Path]:
    """Save drawdown curve plot."""
    if not HAS_MPL:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.size == 0:
        logger.warning("Empty equity curve; skipping drawdown curve plot")
        return None
    eq = np.nan_to_num(eq, nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-12)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(range(len(dd)), dd, alpha=0.4, color="red")
    ax.plot(dd, linewidth=0.6, color="darkred")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=150)
    plt.close(fig)
    return output_path


def plot_return_comparison(
    agent_equity: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Return Comparison",
    fmt: str = "pdf",
) -> Optional[Path]:
    """Overlay equity curves of multiple agents."""
    if not HAS_MPL:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = 0
    for name, eq in agent_equity.items():
        eq_arr = np.asarray(eq, dtype=np.float64)
        if eq_arr.size == 0:
            continue
        eq_arr = np.nan_to_num(eq_arr, nan=0.0, posinf=0.0, neginf=0.0)
        norm = eq_arr / max(eq_arr[0], 1e-12)
        ax.plot(norm, label=name, linewidth=0.8)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        logger.warning("No non-empty curves available; skipping return comparison plot")
        return None
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Normalized Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=150)
    plt.close(fig)
    return output_path


def plot_risk_comparison(
    agent_metrics: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Risk-Adjusted Comparison",
    fmt: str = "pdf",
) -> Optional[Path]:
    """Bar chart comparing Sharpe, Sortino, max drawdown across agents."""
    if not HAS_MPL:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agents = sorted(agent_metrics.keys())
    if not agents:
        logger.warning("No agent metrics provided; skipping risk comparison plot")
        return None
    metric_names = ["sharpe_ratio", "sortino_ratio", "max_drawdown"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 4))
    if len(metric_names) == 1:
        axes = [axes]

    for ax, mn in zip(axes, metric_names):
        vals = [agent_metrics[a].get(mn, 0) for a in agents]
        ax.bar(agents, vals, alpha=0.7)
        ax.set_title(mn.replace("_", " ").title())
        ax.set_ylabel(mn)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=150)
    plt.close(fig)
    return output_path


def plot_drawdown_comparison(
    agent_equity: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Drawdown Comparison",
    fmt: str = "pdf",
) -> Optional[Path]:
    """Overlay drawdown curves for multiple agents."""
    if not HAS_MPL:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = 0
    for name, eq in agent_equity.items():
        eq = np.asarray(eq, dtype=np.float64)
        if len(eq) == 0:
            continue
        eq = np.nan_to_num(eq, nan=0.0, posinf=0.0, neginf=0.0)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.maximum(peak, 1e-12)
        ax.plot(dd, label=name, linewidth=0.8)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        logger.warning("No non-empty curves available; skipping drawdown comparison plot")
        return None

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=150)
    plt.close(fig)
    return output_path
