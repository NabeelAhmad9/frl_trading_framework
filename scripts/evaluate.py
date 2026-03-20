"""CLI entry point — evaluate a trained agent."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import resolve_config
from src.workflows.data_workflow import run_data_workflow
from src.workflows.evaluate_workflow import run_evaluate_workflow
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
    parser = argparse.ArgumentParser(description="Evaluate trained FRL agent")
    parser.add_argument("--agent", default="dqn", help="Agent name (dqn/doubledqn) or agent config path")
    parser.add_argument("--pair", default="EURUSD", help="Currency pair")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--outputs", default="outputs", help="Output root directory")
    args = parser.parse_args()

    config = resolve_config(root=str(PROJECT_ROOT), agent_config=normalize_agent_arg(args.agent))
    data_results = run_data_workflow(config, pairs=[args.pair], root=str(PROJECT_ROOT))
    eval_df = data_results[args.pair][args.split]

    agent_name = config.get("agent", {}).get("name", "dqn")
    from src.utils.artifact_manager import ArtifactManager
    am = ArtifactManager(args.outputs)
    run_dir = am.agent_result_dir(agent_name, args.pair)

    metrics = run_evaluate_workflow(
        config=config, pair=args.pair, eval_df=eval_df,
        run_dir=run_dir, checkpoint_path=args.checkpoint, split_name=args.split,
    )
    logger.info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()
