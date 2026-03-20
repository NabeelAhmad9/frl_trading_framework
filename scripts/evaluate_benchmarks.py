"""CLI entry point — run benchmark strategies across all pairs."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import resolve_config, load_yaml
from src.workflows.data_workflow import run_data_workflow
from src.workflows.benchmark_workflow import run_benchmark_workflow
from src.utils.logger import get_logger
from src.utils.progress import get_progress_colour, progress_iter, progress_logging_redirect, set_bar_metrics

logger = get_logger(__name__)

BENCHMARKS = ["buy_and_hold", "mean_reversion", "momentum", "random_policy"]
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]


def main():
    parser = argparse.ArgumentParser(description="Run FRL benchmark strategies")
    parser.add_argument("--benchmarks", nargs="*", default=BENCHMARKS)
    parser.add_argument("--pairs", nargs="*", default=PAIRS)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()

    config = resolve_config(root=str(PROJECT_ROOT))

    # Build data for all pairs
    try:
        data_results = run_data_workflow(config, pairs=args.pairs, root=str(PROJECT_ROOT))
        data_by_split = {
            "train": {
                pair: data_results[pair]["train"]
                for pair in args.pairs
                if pair in data_results and "train" in data_results[pair]
            },
            "test": {
                pair: data_results[pair]["test"]
                for pair in args.pairs
                if pair in data_results and "test" in data_results[pair]
            },
        }
    except Exception as e:
        logger.error("Data loading failed: %s", e)
        return

    with progress_logging_redirect(config):
        bench_pbar = progress_iter(
            args.benchmarks,
            config,
            desc="Benchmarks",
            total=len(args.benchmarks),
            position=0,
            leave=True,
            unit="bench",
            colour=get_progress_colour(config, "benchmark"),
        )

        for bench_name in bench_pbar:
            bench_yaml_path = PROJECT_ROOT / "configs" / "benchmarks" / f"{bench_name}.yaml"
            bench_cfg = load_yaml(bench_yaml_path) if bench_yaml_path.exists() else {}

            split_results = {}
            for split_name in ("train", "test"):
                split_results[split_name] = run_benchmark_workflow(
                    config=config,
                    benchmark_name=bench_name,
                    benchmark_config=bench_cfg,
                    pairs=args.pairs,
                    data=data_by_split.get(split_name, {}),
                    outputs_root=args.outputs,
                    split_name=split_name,
                    progress_position_offset=1,
                )

            set_bar_metrics(
                bench_pbar,
                metrics={
                    "benchmark": bench_name,
                    "train_pairs": len(split_results["train"]),
                    "test_pairs": len(split_results["test"]),
                },
                refresh=False,
            )
            logger.info(
                "Benchmark %s train results: %s",
                bench_name,
                {p: m.get("cumulative_return", None) for p, m in split_results["train"].items()},
            )
            logger.info(
                "Benchmark %s test results: %s",
                bench_name,
                {p: m.get("cumulative_return", None) for p, m in split_results["test"].items()},
            )


if __name__ == "__main__":
    main()
