# FRL Trading Framework

Forex Reinforcement Learning Trading Framework — a modular, research-grade system for training and evaluating DQN and Double DQN agents on hourly Forex data.

## Scope

- **Pairs**: EURUSD, GBPUSD, USDJPY, AUDUSD (hourly bars)
- **Agents**: DQN, Double DQN
- **Actions**: 10-action extended space (hold, open, pyramid, martingale, reduce, close, reverse) + simplified 3-action adapter
- **Experiments**: EURUSD-only, Double DQN-only ablation families
- **Final Evaluation**: Both agents on all four pairs + benchmark comparisons

## Raw Data

Place hourly OHLCV CSV files in `data/raw/`:
```
data/raw/EURUSD.csv
data/raw/GBPUSD.csv
data/raw/USDJPY.csv
data/raw/AUDUSD.csv
```

Each CSV must contain: `timestamp, open, high, low, close, volume`

## Commands

### Build Datasets
```bash
python scripts/build_dataset.py                    # all pairs
python scripts/build_dataset.py --pair EURUSD      # single pair
```

### Train
```bash
python scripts/train.py --agent dqn --pair EURUSD
python scripts/train.py --agent doubledqn --pair EURUSD
```

### Evaluate
```bash
python scripts/evaluate.py --agent dqn --pair EURUSD
```

### Run Experiments
```bash
python scripts/run_experiment.py --experiment 01_reward_ablation
```

### Benchmarks
```bash
python scripts/evaluate_benchmarks.py
```

### Compare Agents
```bash
python scripts/compare_agents.py
```

## Anti-Lookahead Execution Timing

The canonical timing enforced everywhere:
1. **Observe** through `close_t`
2. **Decide** at time `t`
3. **Execute** at `open_{t+1}` with costs (spread → slippage → commission → rollover)
4. **Mark-to-market** at `close_{t+1}` for reward

## Artifact Layout

All outputs go to `outputs/` via `ArtifactManager`:
```
outputs/
├── experiments/{family}/{variant}/    # experiment runs
├── results/agents/{agent}/{pair}/     # final evaluation
├── results/benchmarks/{bench}/{pair}/ # benchmark results
└── results/comparisons/              # DQN vs Double DQN
```

Each run directory contains: `resolved_config.yaml`, `checkpoints/`, `models/`, `metrics/`, `figures/`, `tables/`, `logs/`

## Experiment Policy

- **Experiment families**: EURUSD only, Double DQN only
- **Differences**: YAML config overrides only — no Python branching
- **Final evaluation**: DQN + Double DQN on all 4 pairs

## Configuration

All runtime behavior is driven by YAML files under `configs/`. Config merge order:
1. `configs/base/*.yaml` (data → features → environment → reward → training → evaluation → logging)
2. `configs/agents/{agent}.yaml`
3. Optional experiment/benchmark overrides
4. CLI overrides

Random seed: 42 (default baseline).

### Progress Bars

Training, experiment, benchmark, and workflow loops use terminal progress bars
via `tqdm` with nested, color-coded rendering.

Controls are under `logging.progress` in `configs/base/logging.yaml`:

- `enabled`: global on/off switch
- `verbosity`: `quiet` | `normal` | `verbose`
- `refresh_rate`: min refresh interval (seconds)
- `metrics_update_interval`: how often inline metrics are refreshed
- `colours`: per-loop color mapping (training, episode, experiment, evaluation, benchmark, pair, agent)

Set `logging.progress.enabled: false` to disable bars globally while preserving
all existing logging/checkpoint/metric outputs.

## Project Structure

- `src/data/` — Data loading, preprocessing, dataset abstraction
- `src/features/` — Technical and microstructure feature engineering
- `src/environment/` — Trading environment with market, portfolio, state management
- `src/reward/` — Decomposable, config-driven reward components
- `src/models/` — Neural network encoders and heads
- `src/agents/` — DQN and Double DQN with replay buffer
- `src/training/` — Training loop, checkpointing, scheduling
- `src/evaluation/` — Backtesting, metrics, reports, visualization
- `src/experiments/` — Experiment registry, variant builder, runner
- `src/benchmarks/` — Baseline strategies (buy & hold, momentum, mean reversion, random)
- `src/workflows/` — Thin orchestration layer
- `src/utils/` — Config loading, logging, device, seed, timers, artifact management
- `notebooks/core/` — Data, feature, environment, reward, training, and results notebooks
- `notebooks/experiments/` — Per-experiment-family analysis notebooks

## Notebooks

### Core Notebooks
```
notebooks/core/
├── 01_data_processing.ipynb        # Raw data quality, splits, missing values
├── 02_feature_engineering.ipynb    # Feature distributions, correlations, warm-up trimming
├── 03_environment_validation.ipynb # Deterministic episodes, legal masks, fills
├── 04_reward_analysis.ipynb        # Reward decomposition, component sensitivity
├── 05_training_runs.ipynb          # Training curves, loss, checkpoint inspection
└── 06_results_visualization.ipynb  # Publication-ready figures from saved outputs
```

### Experiment Notebooks
```
notebooks/experiments/
├── 01_reward_ablation_analysis.ipynb
├── 02_action_space_analysis.ipynb
├── 03_scaling_analysis.ipynb
├── 04_transaction_cost_robustness_analysis.ipynb
└── 05_slippage_robustness_analysis.ipynb
```

All notebooks read from saved artifacts under `outputs/` — they do not re-implement production logic.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific phase
pytest tests/data/ -v
pytest tests/features/ -v
pytest tests/environment/ -v
pytest tests/agents/ -v
pytest tests/evaluation/ -v
pytest tests/benchmarks/ -v
pytest tests/workflows/ -v

# Run smoke tests only
pytest tests/smoke/ -v
```

