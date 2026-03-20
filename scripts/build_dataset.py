"""Build processed datasets from raw CSV files.

Usage:
    python scripts/build_dataset.py [--pair EURUSD] [--config path/to/override.yaml]
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import resolve_config
from src.utils.logger import setup_logger
from src.workflows.data_workflow import run_data_workflow


def main():
    parser = argparse.ArgumentParser(description="Build processed Forex datasets")
    parser.add_argument("--pair", type=str, default=None, help="Single pair to process (default: all)")
    parser.add_argument("--config", type=str, default=None, help="Optional config override YAML")
    args = parser.parse_args()

    logger = setup_logger("build_dataset", level="INFO")

    config = resolve_config(str(REPO_ROOT))
    pairs = [args.pair] if args.pair else None
    run_data_workflow(config, pairs=pairs, root=str(REPO_ROOT))
    logger.info("Dataset build complete.")


if __name__ == "__main__":
    main()
