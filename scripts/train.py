"""CLI entry point — train an agent on one pair."""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import resolve_config
from src.workflows.data_workflow import run_data_workflow
from src.workflows.train_workflow import run_train_workflow
from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_agent_arg(agent: str) -> str:
    """Accept short agent names or explicit YAML paths."""
    candidate = (agent or "dqn").strip()
    normalized = candidate.lower()
    if normalized in {"dqn", "doubledqn"}:
        return f"configs/agents/{normalized}.yaml"
    if candidate.endswith(".yaml"):
        return candidate
    return f"configs/agents/{candidate}.yaml"


def main():
    parser = argparse.ArgumentParser(description="Train FRL agent")
    parser.add_argument("--agent", default="dqn", help="Agent name (dqn/doubledqn) or agent config path")
    parser.add_argument("--pair", default="EURUSD", help="Currency pair")
    parser.add_argument("--outputs", default="outputs", help="Output root directory")
    parser.add_argument("--experiment", nargs="*", default=None, help="Experiment override YAMLs")
    args = parser.parse_args()

    config = resolve_config(
        root=str(PROJECT_ROOT),
        agent_config=normalize_agent_arg(args.agent),
        experiment_overrides=args.experiment,
    )

    # Build data
    data_results = run_data_workflow(config, pairs=[args.pair], root=str(PROJECT_ROOT))
    train_df = data_results[args.pair]["train"]

    summary = run_train_workflow(
        config=config,
        pair=args.pair,
        train_df=train_df,
        outputs_root=args.outputs,
    )
    logger.info("Done: %s", summary)


if __name__ == "__main__":
    main()
