"""Evaluate workflow — run evaluation, compute metrics, generate reports."""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.evaluation.backtester import run_backtest
from src.evaluation.metrics import compute_metrics
from src.evaluation.performance_report import (
    save_actions_sequence,
    save_drawdown_curve,
    save_equity_curve,
    save_metric_snapshots,
    save_performance_summary,
    save_risk_metrics,
    save_trade_log,
)
from src.evaluation.visualization import plot_equity_curve, plot_drawdown_curve
from src.utils.artifact_manager import ArtifactManager
from src.utils.config_loader import save_resolved_config
from src.utils.logger import get_logger, write_log_file

logger = get_logger(__name__)


def run_evaluate_workflow(
    config: Dict[str, Any],
    pair: str,
    eval_df: pd.DataFrame,
    run_dir: Path,
    checkpoint_path: Optional[str] = None,
    agent=None,
    split_name: str = "test",
    progress_position_offset: int = 0,
) -> Dict[str, Any]:
    """Evaluate a trained agent and produce canonical artifacts.

    Parameters
    ----------
    config : dict
        Resolved configuration.
    pair : str
    eval_df : pd.DataFrame
        Feature-ready evaluation data.
    run_dir : Path
        Base run directory (e.g. outputs/results/agents/dqn/EURUSD/).
    checkpoint_path : str, optional
    agent : optional
    split_name : str
        ``"train"`` or ``"test"`` — controls sub-directory placement.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    run_dir = Path(run_dir)
    split_name = str(split_name).strip().lower() or "test"
    am = ArtifactManager(run_dir.parent.parent if run_dir.name != "outputs" else run_dir)
    dirs = am.ensure_subdirs(run_dir)

    save_resolved_config(config, am.resolved_config_path(run_dir))

    eval_cfg = config.get("evaluation", {})
    deterministic = eval_cfg.get("deterministic_policy", True)
    periods_per_year = eval_cfg.get("periods_per_year", 6048)
    rf = eval_cfg.get("risk_free_rate", 0.0)
    fig_fmt = eval_cfg.get("figure_format", "pdf")

    result = run_backtest(
        df=eval_df, pair=pair, config=config,
        checkpoint_path=checkpoint_path, agent=agent,
        deterministic=deterministic,
        progress_position_offset=progress_position_offset,
    )

    equity = np.array(result.equity_curve)
    metrics = compute_metrics(
        equity, trade_log=result.trade_log,
        periods_per_year=periods_per_year, risk_free_rate=rf,
    )

    # Save artifacts under split-specific dirs
    metrics_dir = dirs.get(f"metrics_{split_name}", run_dir / "metrics" / split_name)
    tables_dir = dirs.get(f"tables_{split_name}", run_dir / "tables" / split_name)
    figures_dir = dirs.get(f"figures_{split_name}", run_dir / "figures" / split_name)
    logs_dir = dirs.get("logs", run_dir / "logs")

    save_metric_snapshots(metrics, metrics_dir)
    save_actions_sequence(result.actions, metrics_dir / "actions_sequence.csv")
    save_trade_log(result.trade_log, metrics_dir / "trade_log.csv")
    save_performance_summary(metrics, tables_dir / "performance_summary.csv")
    save_risk_metrics(metrics, tables_dir / "risk_metrics.csv")
    save_equity_curve(equity, metrics_dir / "equity_curve.csv")
    save_drawdown_curve(equity, metrics_dir / "drawdown_curve.csv")
    plot_equity_curve(equity, figures_dir / f"equity_curve.{fig_fmt}", title=f"{pair} Equity ({split_name})", fmt=fig_fmt)
    plot_drawdown_curve(equity, figures_dir / f"drawdown_curve.{fig_fmt}", title=f"{pair} Drawdown ({split_name})", fmt=fig_fmt)
    write_log_file(
        logs_dir / "evaluation.log",
        [
            f"evaluation_complete pair={pair} split={split_name}",
            f"cumulative_return={metrics['cumulative_return']:.6f}",
            f"sharpe_ratio={metrics['sharpe_ratio']:.6f}",
            f"max_drawdown={metrics['max_drawdown']:.6f}",
            f"turnover={metrics['turnover']:.6f}",
        ],
        mode="w",
    )

    logger.info("Evaluation complete for %s (%s): cumret=%.4f sharpe=%.2f dd=%.4f",
                pair, split_name, metrics["cumulative_return"],
                metrics["sharpe_ratio"], metrics["max_drawdown"])

    return metrics
