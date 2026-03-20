"""CLI entry point — compare DQN vs Double DQN across all pairs."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.backtester import run_backtest
from src.evaluation.performance_report import (
    save_comparison_table,
    save_drawdown_curve,
    save_equity_curve,
    save_statistical_tests,
)
from src.evaluation.statistical_tests import compare_agents as stat_compare
from src.evaluation.visualization import (
    plot_drawdown_comparison,
    plot_return_comparison,
    plot_risk_comparison,
)
from src.utils.artifact_manager import ArtifactManager
from src.utils.config_loader import resolve_config
from src.workflows.data_workflow import run_data_workflow
from src.workflows.evaluate_workflow import run_evaluate_workflow
from src.utils.logger import get_logger, write_log_file
from src.utils.progress import get_progress_colour, progress_iter, progress_logging_redirect, set_bar_metrics

logger = get_logger(__name__)

AGENTS = ["dqn", "doubledqn"]
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]


def _equity_to_returns(equity_curve):
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    if len(equity_curve) < 2:
        return np.array([], dtype=np.float64)
    return np.diff(equity_curve) / np.maximum(np.abs(equity_curve[:-1]), 1e-12)


def _aggregate_equity(curves):
    if not curves:
        return np.array([], dtype=np.float64)
    min_len = min(len(curve) for curve in curves)
    trimmed = [np.asarray(curve[:min_len], dtype=np.float64) for curve in curves]
    normalized = [curve / max(curve[0], 1e-12) for curve in trimmed]
    return np.mean(np.stack(normalized, axis=0), axis=0)


def _mean_metrics(metric_map):
    keys = set()
    for metrics in metric_map.values():
        keys.update(metrics.keys())
    return {
        key: float(np.mean([metrics[key] for metrics in metric_map.values() if key in metrics]))
        for key in sorted(keys)
    }


def main():
    parser = argparse.ArgumentParser(description="Compare DQN vs Double DQN")
    parser.add_argument("--outputs", default="outputs", help="Outputs root")
    parser.add_argument("--pairs", nargs="*", default=PAIRS)
    args = parser.parse_args()

    am = ArtifactManager(args.outputs)
    progress_cfg = resolve_config(root=str(PROJECT_ROOT), agent_config="configs/agents/dqn.yaml")

    all_metrics = {}  # {agent: {pair: metrics}}
    all_equity = {}   # {agent: {pair: equity_array}}

    with progress_logging_redirect(progress_cfg):
        agent_pbar = progress_iter(
            AGENTS,
            progress_cfg,
            desc="Agents",
            total=len(AGENTS),
            position=0,
            leave=True,
            unit="agent",
            colour=get_progress_colour(progress_cfg, "agent"),
        )

        for agent_name in agent_pbar:
            all_metrics[agent_name] = {}
            all_equity[agent_name] = {}
            agent_yaml = f"configs/agents/{agent_name}.yaml"

            pair_pbar = progress_iter(
                args.pairs,
                progress_cfg,
                desc=f"{agent_name} pairs",
                total=len(args.pairs),
                position=1,
                leave=False,
                unit="pair",
                colour=get_progress_colour(progress_cfg, "pair"),
            )

            for pair in pair_pbar:
                config = resolve_config(root=str(PROJECT_ROOT), agent_config=agent_yaml)
                run_dir = am.agent_result_dir(agent_name, pair)
                ckpt_path = run_dir / "checkpoints" / "checkpoint_latest.pt"

                if not ckpt_path.exists():
                    logger.warning("No checkpoint for %s/%s — skipping", agent_name, pair)
                    continue

                # Load train data
                try:
                    data_results = run_data_workflow(config, pairs=[pair], root=str(PROJECT_ROOT))
                    train_df = data_results[pair]["train"]
                except Exception as e:
                    logger.warning("Data load failed for %s: %s", pair, e)
                    continue

                metrics = run_evaluate_workflow(
                    config=config, pair=pair, eval_df=train_df,
                    run_dir=run_dir, checkpoint_path=str(ckpt_path), split_name="train",
                    progress_position_offset=2,
                )
                all_metrics[agent_name][pair] = metrics
                backtest = run_backtest(
                    df=train_df,
                    pair=pair,
                    config=config,
                    checkpoint_path=str(ckpt_path),
                    deterministic=config.get("evaluation", {}).get("deterministic_policy", True),
                    progress_position_offset=2,
                )
                all_equity[agent_name][pair] = np.asarray(backtest.equity_curve, dtype=np.float64)
                set_bar_metrics(
                    pair_pbar,
                    metrics={
                        "pair": pair,
                        "cumret": f"{metrics.get('cumulative_return', 0.0):.4f}",
                    },
                    refresh=False,
                )

            set_bar_metrics(
                agent_pbar,
                metrics={
                    "agent": agent_name,
                    "pairs": len(all_metrics.get(agent_name, {})),
                },
                refresh=False,
            )

    # By-pair comparisons
    with progress_logging_redirect(progress_cfg):
        pair_summary_pbar = progress_iter(
            args.pairs,
            progress_cfg,
            desc="Pair comparisons",
            total=len(args.pairs),
            position=0,
            leave=True,
            unit="pair",
            colour=get_progress_colour(progress_cfg, "experiment"),
        )
        for pair in pair_summary_pbar:
            pair_metrics = {}
            pair_equity = {}
            for agent_name in AGENTS:
                if pair in all_metrics.get(agent_name, {}):
                    pair_metrics[agent_name] = all_metrics[agent_name][pair]
                if pair in all_equity.get(agent_name, {}):
                    pair_equity[agent_name] = all_equity[agent_name][pair]

            if len(pair_metrics) >= 2:
                pair_dir = am.comparison_by_pair_dir(pair)
                pair_dirs = ArtifactManager.ensure_subdirs(pair_dir, subdirs=["figures", "tables", "logs"])
                pair_returns = {
                    agent_name: _equity_to_returns(equity)
                    for agent_name, equity in pair_equity.items()
                    if len(equity) >= 2
                }
                stats = stat_compare(pair_returns)

                save_comparison_table(pair_metrics, pair_dirs["tables"] / "performance_summary.csv")
                save_statistical_tests(stats, pair_dirs["tables"] / "statistical_tests.csv")

                # Save raw curves for each agent in this pair comparison
                for agent_name, equity in pair_equity.items():
                    agent_pair_metrics_dir = pair_dirs["tables"] / agent_name
                    save_equity_curve(equity, agent_pair_metrics_dir / "equity_curve.csv")
                    save_drawdown_curve(equity, agent_pair_metrics_dir / "drawdown_curve.csv")

                plot_return_comparison(pair_equity, pair_dirs["figures"] / "return_comparison.pdf")
                plot_risk_comparison(pair_metrics, pair_dirs["figures"] / "risk_adjusted_comparison.pdf")
                plot_drawdown_comparison(pair_equity, pair_dirs["figures"] / "drawdown_comparison.pdf")
                write_log_file(
                    pair_dirs["logs"] / "comparison.log",
                    [
                        f"pair={pair}",
                        f"agents={','.join(sorted(pair_metrics.keys()))}",
                    ],
                    mode="w",
                )
                set_bar_metrics(
                    pair_summary_pbar,
                    metrics={
                        "pair": pair,
                        "agents": len(pair_metrics),
                    },
                    refresh=False,
                )

    # All-pairs aggregate
    agg_metrics = {}
    agg_equity = {}
    agg_returns = {}
    with progress_logging_redirect(progress_cfg):
        agg_pbar = progress_iter(
            AGENTS,
            progress_cfg,
            desc="Aggregate metrics",
            total=len(AGENTS),
            position=0,
            leave=True,
            unit="agent",
            colour=get_progress_colour(progress_cfg, "agent"),
        )
        for agent_name in agg_pbar:
            if all_metrics.get(agent_name):
                agg_metrics[agent_name] = _mean_metrics(all_metrics[agent_name])
            curves = [all_equity[agent_name][pair] for pair in args.pairs if pair in all_equity.get(agent_name, {})]
            if curves:
                agg_equity[agent_name] = _aggregate_equity(curves)
                agg_returns[agent_name] = np.concatenate([
                    _equity_to_returns(curve) for curve in curves if len(curve) >= 2
                ])
            set_bar_metrics(
                agg_pbar,
                metrics={
                    "agent": agent_name,
                    "curves": len(curves),
                },
                refresh=False,
            )

    if len(agg_metrics) >= 2:
        all_dir = am.comparison_all_pairs_dir()
        all_dirs = ArtifactManager.ensure_subdirs(all_dir, subdirs=["figures", "tables", "logs"])
        stats = stat_compare(agg_returns)
        save_comparison_table(agg_metrics, all_dirs["tables"] / "performance_summary.csv")
        save_statistical_tests(stats, all_dirs["tables"] / "statistical_tests.csv")

        # Save aggregated raw curves for the final summary
        for agent_name, equity in agg_equity.items():
            agent_agg_metrics_dir = all_dirs["tables"] / agent_name
            save_equity_curve(equity, agent_agg_metrics_dir / "equity_curve.csv")
            save_drawdown_curve(equity, agent_agg_metrics_dir / "drawdown_curve.csv")

        plot_return_comparison(agg_equity, all_dirs["figures"] / "return_comparison.pdf")
        plot_risk_comparison(agg_metrics, all_dirs["figures"] / "risk_adjusted_comparison.pdf")
        plot_drawdown_comparison(agg_equity, all_dirs["figures"] / "drawdown_comparison.pdf")
        write_log_file(
            all_dirs["logs"] / "comparison.log",
            [
                f"pairs={','.join(args.pairs)}",
                f"agents={','.join(sorted(agg_metrics.keys()))}",
            ],
            mode="w",
        )

    logger.info("Comparison complete.")


if __name__ == "__main__":
    main()
