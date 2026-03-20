"""CLI entry point — run all variants of an experiment family."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.workflows.data_workflow import run_data_workflow
from src.workflows.experiment_workflow import run_experiment_workflow
from src.utils.config_loader import resolve_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run an experiment family")
    parser.add_argument("--experiment", required=True, help="Experiment name, e.g. 01_reward_ablation")
    parser.add_argument("--outputs", default="outputs", help="Output root directory")
    args = parser.parse_args()

    # Build EURUSD data with base config (no agent-specific overrides needed for data)
    base_config = resolve_config(root=str(PROJECT_ROOT))
    data_results = run_data_workflow(base_config, pairs=["EURUSD"], root=str(PROJECT_ROOT))
    train_df = data_results["EURUSD"]["train"]
    test_df = data_results["EURUSD"].get("test")

    results = run_experiment_workflow(
        root=PROJECT_ROOT,
        experiment_name=args.experiment,
        train_df=train_df,
        test_df=test_df,
        outputs_root=args.outputs,
    )
    logger.info("Experiment %s complete: %d variants", args.experiment, len(results))


if __name__ == "__main__":
    main()
