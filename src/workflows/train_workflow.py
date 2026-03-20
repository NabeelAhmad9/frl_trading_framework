"""Train workflow — orchestrate a single training run from config."""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.training.trainer import Trainer
from src.utils.artifact_manager import ArtifactManager
from src.utils.config_loader import resolve_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_train_workflow(
    config: Dict[str, Any],
    pair: str,
    train_df: pd.DataFrame,
    outputs_root: str = "outputs",
    run_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one full training loop for a single agent and pair.

    Parameters
    ----------
    config : dict
        Fully resolved configuration.
    pair : str
        Currency pair, e.g. ``EURUSD``.
    train_df : pd.DataFrame
        Feature-ready training data.
    outputs_root : str
        Root directory for artifacts.
    run_tag : str, optional
        Additional tag for the run directory name.

    Returns
    -------
    dict
        Training summary.
    """
    am = ArtifactManager(outputs_root)
    agent_name = config.get("agent", {}).get("name", "dqn")

    run_dir = am.agent_result_dir(agent_name, pair)
    if run_tag:
        run_dir = run_dir / run_tag

    run_dir.mkdir(parents=True, exist_ok=True)
    am.setup_run(run_dir)

    logger.info("Train workflow: agent=%s, pair=%s, run_dir=%s", agent_name, pair, run_dir)

    trainer = Trainer(df=train_df, pair=pair, config=config, run_dir=run_dir)
    summary = trainer.train()

    logger.info("Training complete: %s", summary)
    return summary
