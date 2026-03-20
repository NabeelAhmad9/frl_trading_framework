"""Experiment workflow — orchestrate a full experiment family run."""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.experiments.runner import run_experiment
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_experiment_workflow(
    root: Path,
    experiment_name: str,
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    outputs_root: str = "outputs",
) -> Dict[str, Any]:
    """High-level entry point for running an experiment family.

    Parameters
    ----------
    root : Path
        Repository root.
    experiment_name : str
        Experiment family name (e.g. ``01_reward_ablation``).
    train_df : pd.DataFrame
        Training data.
    test_df : pd.DataFrame, optional
        Test data for evaluation pass.
    outputs_root : str
        Base outputs directory.

    Returns
    -------
    dict
        Per-variant results.
    """
    logger.info("Starting experiment workflow: %s", experiment_name)
    results = run_experiment(
        root=root,
        experiment_name=experiment_name,
        train_df=train_df,
        test_df=test_df,
        outputs_root=outputs_root,
    )
    logger.info("Experiment workflow finished: %s (%d variants)", experiment_name, len(results))
    return results
