"""Benchmark workflow — run benchmark strategies through the evaluation pipeline."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.benchmarks.base_benchmark import BaseBenchmark
from src.benchmarks.buy_and_hold import BuyAndHoldBenchmark
from src.benchmarks.mean_reversion import MeanReversionBenchmark
from src.benchmarks.momentum import MomentumBenchmark
from src.benchmarks.random_policy import RandomPolicyBenchmark
from src.environment.registry import make_env
from src.evaluation.metrics import compute_metrics
from src.evaluation.trade_log import build_trade_event
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
from src.reward.reward_factory import build_reward_engine
from src.utils.artifact_manager import ArtifactManager
from src.utils.config_loader import deep_merge, save_resolved_config
from src.utils.logger import get_logger, write_log_file
from src.utils.progress import (
    get_progress_colour,
    progress_bar,
    progress_iter,
    progress_logging_redirect,
    set_bar_metrics,
)

logger = get_logger(__name__)

BENCHMARK_CLASSES = {
    "buy_and_hold": BuyAndHoldBenchmark,
    "mean_reversion": MeanReversionBenchmark,
    "momentum": MomentumBenchmark,
    "random_policy": RandomPolicyBenchmark,
}


def _run_single_benchmark(
    benchmark: BaseBenchmark,
    df: pd.DataFrame,
    pair: str,
    config: Dict[str, Any],
    split_name: str = "test",
    progress_position_offset: int = 1,
) -> Dict[str, Any]:
    """Run one benchmark through the environment and collect results."""
    reward_engine = build_reward_engine(config)
    env = make_env(df, pair, config, reward_engine=reward_engine)

    obs, info = env.reset(seed=42)
    benchmark.reset()

    equity_curve = [info.get("equity", info.get("equity_after", 100000.0))]
    trade_log = []
    actions = []
    total_reward = 0.0
    steps = 0
    prev_direction = 0
    progress_cfg = config.get("logging", {}).get("progress", {})
    metrics_update_interval = max(int(progress_cfg.get("metrics_update_interval", 50)), 1)

    start_idx = max(getattr(env, "window_length", 1) - 1, 0)
    sim_length = getattr(env.simulator, "length", 0)
    sim_remaining = max(0, sim_length - 1 - start_idx)
    if getattr(env, "max_steps", None) is not None:
        sim_remaining = min(sim_remaining, int(env.max_steps))
    total_steps_hint = sim_remaining if sim_remaining > 0 else None

    done = False

    with progress_bar(
        config,
        desc=f"{benchmark.name}:{pair}:{split_name}",
        total=total_steps_hint,
        initial=0,
        position=int(progress_position_offset),
        leave=False,
        unit="step",
        colour=get_progress_colour(config, "benchmark"),
    ) as bench_pbar:
        while not done:
            action = benchmark.act(obs, info, steps)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            bench_pbar.update(1)
            equity_curve.append(info.get("equity", info.get("equity_after", equity_curve[-1])))
            actions.append({
                "timestamp": info.get("timestamp_decision", ""),
                "action_id": info.get("raw_action", action),
                "action_name": info.get("action_name", ""),
                "was_legal": info.get("was_legal", True),
                "executed_action_name": info.get("executed_action_name", info.get("action_name", "")),
                "direction_before": info.get("direction_before", prev_direction),
                "direction_after": info.get("direction", 0),
                "total_lots_after": info.get("total_lots", 0),
            })

            trade_event = build_trade_event(info, step=steps)
            if trade_event is not None:
                trade_log.append(trade_event)
            prev_direction = info.get("direction", prev_direction)

            if (steps % metrics_update_interval) == 0 or done:
                set_bar_metrics(
                    bench_pbar,
                    metrics={
                        "reward": f"{total_reward:.2f}",
                        "equity": f"{equity_curve[-1]:.2f}",
                    },
                    refresh=False,
                )

    eq = np.array(equity_curve)
    eval_cfg = config.get("evaluation", {})
    periods_per_year = eval_cfg.get("periods_per_year", 6048)
    rf = eval_cfg.get("risk_free_rate", 0.0)

    metrics = compute_metrics(eq, trade_log=trade_log, periods_per_year=periods_per_year, risk_free_rate=rf)
    return {
        "metrics": metrics,
        "equity_curve": eq,
        "steps": steps,
        "total_reward": total_reward,
        "actions": actions,
        "trade_log": trade_log,
    }


def run_benchmark_workflow(
    config: Dict[str, Any],
    benchmark_name: str,
    benchmark_config: Dict[str, Any],
    pairs: List[str],
    data: Dict[str, pd.DataFrame],
    outputs_root: str = "outputs",
    split_name: str = "test",
    progress_position_offset: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Run one benchmark across multiple pairs.

    Parameters
    ----------
    config : dict
        Base resolved config.
    benchmark_name : str
    benchmark_config : dict
        Benchmark-specific YAML config (merged).
    pairs : list of str
    data : dict
        pair → DataFrame.
    outputs_root : str
    split_name : str

    Returns
    -------
    dict
        pair → metrics.
    """
    am = ArtifactManager(outputs_root)
    fig_fmt = config.get("evaluation", {}).get("figure_format", "pdf")

    # Merge benchmark config into base config
    merged_cfg = deep_merge(config, benchmark_config or {})
    merged_cfg = deep_merge(merged_cfg, {"benchmark": {"name": benchmark_name}})

    cls = BENCHMARK_CLASSES.get(benchmark_name)
    if cls is None:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    results = {}
    with progress_logging_redirect(merged_cfg):
        pair_pbar = progress_iter(
            pairs,
            merged_cfg,
            desc=f"Benchmark {benchmark_name} [{split_name}]",
            total=len(pairs),
            position=int(progress_position_offset),
            leave=True,
            unit="pair",
            colour=get_progress_colour(merged_cfg, "benchmark"),
        )

        for pair in pair_pbar:
            if pair not in data:
                logger.warning("No data for pair %s — skipping", pair)
                continue

            benchmark = cls(merged_cfg)
            df = data[pair]
            run_result = _run_single_benchmark(
                benchmark,
                df,
                pair,
                merged_cfg,
                split_name=split_name,
                progress_position_offset=int(progress_position_offset) + 1,
            )
            metrics = run_result["metrics"]
            eq = run_result["equity_curve"]

            # Save artifacts
            run_dir = am.benchmark_result_dir(benchmark_name, pair)
            dirs = am.ensure_subdirs(run_dir)
            save_resolved_config(merged_cfg, am.resolved_config_path(run_dir))

            metrics_dir = dirs.get(f"metrics_{split_name}", run_dir / "metrics" / split_name)
            tables_dir = dirs.get(f"tables_{split_name}", run_dir / "tables" / split_name)
            figures_dir = dirs.get(f"figures_{split_name}", run_dir / "figures" / split_name)
            logs_dir = dirs.get("logs", run_dir / "logs")

            save_metric_snapshots(metrics, metrics_dir)
            save_actions_sequence(run_result.get("actions", []), metrics_dir / "actions_sequence.csv")
            save_trade_log(run_result.get("trade_log", []), metrics_dir / "trade_log.csv")
            save_performance_summary(metrics, tables_dir / "performance_summary.csv")
            save_risk_metrics(metrics, tables_dir / "risk_metrics.csv")
            save_equity_curve(eq, metrics_dir / "equity_curve.csv")
            save_drawdown_curve(eq, metrics_dir / "drawdown_curve.csv")
            plot_equity_curve(eq, figures_dir / f"equity_curve.{fig_fmt}", title=f"{benchmark_name} {pair}", fmt=fig_fmt)
            plot_drawdown_curve(eq, figures_dir / f"drawdown_curve.{fig_fmt}", title=f"{benchmark_name} {pair} DD", fmt=fig_fmt)
            write_log_file(
                logs_dir / "evaluation.log",
                [
                    f"benchmark_complete benchmark={benchmark_name} pair={pair} split={split_name}",
                    f"cumulative_return={metrics['cumulative_return']:.6f}",
                    f"sharpe_ratio={metrics['sharpe_ratio']:.6f}",
                    f"max_drawdown={metrics['max_drawdown']:.6f}",
                    f"turnover={metrics['turnover']:.6f}",
                ],
                mode="w",
            )

            results[pair] = metrics
            set_bar_metrics(
                pair_pbar,
                metrics={
                    "pair": pair,
                    "cumret": f"{metrics.get('cumulative_return', 0.0):.4f}",
                    "sharpe": f"{metrics.get('sharpe_ratio', 0.0):.2f}",
                },
                refresh=False,
            )
            logger.info("Benchmark %s/%s [%s]: cumret=%.4f sharpe=%.2f",
                        benchmark_name, pair, split_name, metrics["cumulative_return"], metrics["sharpe_ratio"])

    return results