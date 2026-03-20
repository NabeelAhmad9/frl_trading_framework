"""Experiment runner — execute training and evaluation for each variant."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.experiments.registry import get_experiment
from src.experiments.variant_builder import resolve_all_variants
from src.evaluation.performance_report import save_performance_summary, save_comparison_table
from src.evaluation.visualization import plot_risk_comparison
from src.training.trainer import Trainer
from src.utils.artifact_manager import ArtifactManager
from src.utils.config_loader import save_resolved_config
from src.utils.logger import get_logger, write_log_file
from src.utils.progress import (
    get_progress_colour,
    progress_iter,
    progress_logging_redirect,
    set_bar_metrics,
)
from src.workflows.evaluate_workflow import run_evaluate_workflow

logger = get_logger(__name__)


def run_experiment(
    root: Path,
    experiment_name: str,
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    outputs_root: str = "outputs",
) -> Dict[str, Any]:
    """Run all variants of an experiment family.

    Parameters
    ----------
    root : Path
        Repository root.
    experiment_name : str
        e.g. ``01_reward_ablation``.
    train_df : pd.DataFrame
        Training data.
    test_df : pd.DataFrame, optional
        Test data for evaluation.
    outputs_root : str

    Returns
    -------
    dict
        variant_name → summary dict.
    """
    root = Path(root)
    experiments_root = root / "configs" / "experiments"
    experiment = get_experiment(experiments_root, experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    am = ArtifactManager(outputs_root)
    pair = experiment["metadata"].get("base_pair", "EURUSD")

    variants = resolve_all_variants(root, experiment)
    results = {}

    progress_cfg = variants[0]["config"] if variants else {}
    with progress_logging_redirect(progress_cfg):
        variant_pbar = progress_iter(
            variants,
            progress_cfg,
            desc=f"Experiment {experiment_name}",
            total=len(variants),
            position=0,
            leave=True,
            unit="variant",
            colour=get_progress_colour(progress_cfg, "experiment"),
        )

        for variant in variant_pbar:
            vname = variant["name"]
            config = variant["config"]

            logger.info("=" * 60)
            logger.info("Running variant: %s/%s", experiment_name, vname)

            # Per-variant output directory
            variant_dir = am.experiment_variant_dir(experiment_name, vname)
            dirs = am.setup_run(variant_dir)

            # Save resolved config before execution
            save_resolved_config(config, am.resolved_config_path(variant_dir))

            # Train
            trainer = Trainer(
                df=train_df,
                pair=pair,
                config=config,
                run_dir=variant_dir,
                progress_position_offset=1,
            )
            train_summary = trainer.train()

            # Evaluate on test split if available
            eval_metrics = {}
            if test_df is not None:
                eval_metrics = run_evaluate_workflow(
                    config=config,
                    pair=pair,
                    eval_df=test_df,
                    run_dir=variant_dir,
                    checkpoint_path=str(variant_dir / "checkpoints" / "checkpoint_latest.pt"),
                    split_name="test",
                    progress_position_offset=1,
                )

            results[vname] = {
                "train_summary": train_summary,
                "test_metrics": eval_metrics,
            }
            set_bar_metrics(
                variant_pbar,
                metrics={
                    "variant": vname,
                    "steps": train_summary.get("total_steps", 0),
                    "best_rew": f"{train_summary.get('best_episode_reward', 0.0):.2f}",
                },
                refresh=False,
            )

    # Generate summary
    summary_dir = am.experiment_summary_dir(experiment_name)
    _generate_summary(results, summary_dir)

    logger.info("Experiment %s complete: %d variants", experiment_name, len(results))
    return results


def _generate_summary(results: Dict[str, Any], summary_dir: Path) -> None:
    """Aggregate variant results into summary tables."""
    summary_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = summary_dir / "tables"
    figures_dir = summary_dir / "figures"
    logs_dir = summary_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Collect test metrics if available
    agent_metrics = {}
    for vname, data in results.items():
        tm = data.get("test_metrics", {})
        if tm:
            agent_metrics[vname] = tm

    if agent_metrics:
        save_comparison_table(agent_metrics, tables_dir / "performance_summary.csv")
        plot_risk_comparison(agent_metrics, figures_dir / "risk_adjusted_comparison.pdf")

    # Save training summaries
    import csv
    train_path = tables_dir / "training_summary.csv"
    if results:
        first_keys = list(list(results.values())[0].get("train_summary", {}).keys())
        with open(train_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["variant"] + first_keys)
            for vname, data in results.items():
                ts = data.get("train_summary", {})
                writer.writerow([vname] + [ts.get(k, "") for k in first_keys])

    write_log_file(
        logs_dir / "summary.log",
        [
            f"variants={len(results)}",
            f"variants_with_test_metrics={len(agent_metrics)}",
        ],
        mode="w",
    )

    logger.info("Summary written to %s", summary_dir)
