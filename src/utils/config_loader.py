"""Configuration loading, merging, validation, and resolution.

This module is the single entry point for all YAML-based config resolution.
It supports deep-merge of multiple config files in a defined order,
validates required keys and value ranges, normalizes important paths,
and saves resolved configs exactly as executed.
"""

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *override* into a copy of *base*.

    - Nested dicts are merged recursively.
    - Lists and scalar values in *override* replace those in *base*.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Write *data* to a YAML file at *path* and return the resolved path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


# ---------------------------------------------------------------------------
# Config merge pipeline
# ---------------------------------------------------------------------------

_BASE_MERGE_ORDER = [
    "configs/base/data.yaml",
    "configs/features/technical.yaml",
    "configs/features/microstructure.yaml",
    "configs/features/pipeline.yaml",
    "configs/base/environment.yaml",
    "configs/base/reward.yaml",
    "configs/base/training.yaml",
    "configs/base/evaluation.yaml",
    "configs/base/logging.yaml",
]

_MISSING = object()


def _get_value(config: Dict[str, Any], key_path: str, default: Any = _MISSING) -> Any:
    """Return a dot-delimited value from *config*.

    Parameters
    ----------
    config : dict
        Source configuration mapping.
    key_path : str
        Dot-delimited path, e.g. ``training.total_timesteps``.
    default : Any
        Optional fallback returned when the path is absent.
    """
    node: Any = config
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(f"Missing config key: {key_path}")
            return default
        node = node[part]
    return node


def _resolve_config_path(root: Path, relative_path: str, label: str) -> Path:
    """Resolve *relative_path* under *root* and fail fast if it does not exist."""
    resolved = root / relative_path
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def validate_nonnegative(config: Dict[str, Any], keys: List[str]) -> None:
    """Raise ``ValueError`` if any value at *keys* is negative."""
    for key_path in keys:
        node = _get_value(config, key_path)
        if node is not None and node < 0:
            raise ValueError(f"Config key {key_path} must be nonnegative, got {node}")


def validate_pairs_have_metadata(config: Dict[str, Any]) -> None:
    """Ensure every configured pair has a corresponding metadata block."""
    pairs = _get_value(config, "data.pairs")
    pair_metadata = _get_value(config, "data.pair_metadata")
    missing = [pair for pair in pairs if pair not in pair_metadata]
    if missing:
        raise KeyError(f"Missing pair metadata for: {missing}")


def validate_training_consistency(config: Dict[str, Any]) -> None:
    """Validate cross-field training settings that must remain internally consistent."""
    total_timesteps = int(_get_value(config, "training.total_timesteps"))
    warmup_steps = int(_get_value(config, "training.warmup_steps", default=0) or 0)
    learn_start = int(_get_value(config, "agent.training.learn_start_steps", default=warmup_steps) or warmup_steps)

    if total_timesteps <= max(warmup_steps, learn_start):
        raise ValueError(
            "training.total_timesteps must exceed both training.warmup_steps and "
            f"agent.training.learn_start_steps; got total_timesteps={total_timesteps}, "
            f"warmup_steps={warmup_steps}, learn_start_steps={learn_start}"
        )

    checkpoint_interval = _get_value(config, "training.checkpoint_interval", default=None)
    evaluation_interval = _get_value(config, "training.evaluation_interval", default=None)
    for name, value in (
        ("training.checkpoint_interval", checkpoint_interval),
        ("training.evaluation_interval", evaluation_interval),
    ):
        if value is not None and int(value) <= 0:
            raise ValueError(f"Config key {name} must be positive when provided, got {value}")


def validate_cost_order(config: Dict[str, Any]) -> None:
    """Validate execution cost ordering against the supported canonical set."""
    allowed = {"spread", "slippage", "commission", "rollover"}
    order = _get_value(config, "environment.execution.apply_cost_order", default=[])
    if not isinstance(order, list):
        raise ValueError("environment.execution.apply_cost_order must be a list")
    unknown = [item for item in order if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown execution cost terms: {unknown}")


def validate_config(config: Dict[str, Any], root: Union[str, Path]) -> Dict[str, Any]:
    """Validate and normalize a resolved configuration, returning a safe copy."""
    root = Path(root)
    config = normalize_paths(config, root, ["data.raw_data_dir", "data.processed_data_dir"])

    config.setdefault("meta", {})
    config["meta"].setdefault("project_name", "frl-trading-framework")
    config["meta"].setdefault("version", "1.0.0")
    config["meta"].setdefault("random_seed", _get_value(config, "training.random_seed", default=42))
    if "agent" in config:
        config["meta"].setdefault("agent_name", _get_value(config, "agent.name", default="unknown"))

    config.setdefault("paths", {})
    config["paths"].setdefault("root", str(root.resolve()))
    config["paths"].setdefault("raw_data_dir", _get_value(config, "data.raw_data_dir"))
    config["paths"].setdefault("processed_data_dir", _get_value(config, "data.processed_data_dir"))

    required = [
        "meta.project_name",
        "meta.version",
        "data.pairs",
        "data.raw_data_dir",
        "data.processed_data_dir",
        "data.timestamp_column",
        "data.ohlcv_columns",
        "data.train_ratio",
        "data.frequency",
        "data.timezone",
        "data.missing_value_policy",
        "data.duplicate_policy",
        "data.output_format",
        "data.pair_metadata",
        "features.pipeline.selected_feature_groups",
        "features.pipeline.normalization.scaler_type",
        "environment.observation.window_length",
        "environment.account.initial_capital",
        "environment.account.currency",
        "environment.actions.base_lot_size",
        "environment.invalid_action.policy",
        "environment.execution.fill_price_rule",
        "environment.execution.mark_price_rule",
        "reward.components.profit.enabled",
        "reward.weights.profit",
        "reward.normalization.method",
        "training.random_seed",
        "training.total_timesteps",
        "training.warmup_steps",
        "training.device.preference",
        "evaluation.deterministic_policy",
        "logging.level",
    ]
    if "agent" in config:
        required.extend([
            "agent.name",
            "agent.algorithm.double_dqn",
            "agent.model.encoder_type",
            "agent.replay.batch_size",
            "agent.training.learn_start_steps",
        ])

    validate_required_keys(config, required)
    validate_positive(config, [
        "environment.observation.window_length",
        "environment.account.initial_capital",
        "environment.leverage.max_leverage",
        "environment.actions.base_lot_size",
        "training.total_timesteps",
    ])
    validate_nonnegative(config, [
        "training.warmup_steps",
        "environment.invalid_action.penalty",
        "environment.transaction_costs.commission_per_lot",
        "environment.transaction_costs.cost_multiplier",
    ])

    train_ratio = float(_get_value(config, "data.train_ratio"))
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"Config key data.train_ratio must be in (0, 1), got {train_ratio}")

    validate_enum(config, "data.output_format", ["parquet"])
    validate_enum(config, "data.duplicate_policy", ["keep_last", "keep_first", "drop"])
    validate_enum(config, "data.missing_value_policy", ["ffill_then_drop_residual", "drop_rows"])
    validate_enum(config, "features.pipeline.normalization.scaler_type", ["standard", "robust", "none"])
    validate_enum(config, "environment.invalid_action.policy", ["convert_to_hold_with_penalty", "hold", "raise"])
    validate_enum(config, "environment.execution.fill_price_rule", ["next_bar_open"])
    validate_enum(config, "environment.execution.mark_price_rule", ["next_bar_close"])
    validate_enum(config, "environment.slippage.mode", ["deterministic", "stochastic"])
    validate_enum(config, "environment.transaction_costs.spread_mode", ["proxy", "fixed"])
    validate_enum(config, "environment.episode.start_policy", ["beginning", "random"])
    validate_enum(config, "environment.episode.end_policy", ["end_of_data", "max_steps"])
    validate_enum(config, "reward.normalization.method", ["clip_only", "none", "running"])
    validate_enum(config, "logging.level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    validate_enum(config, "evaluation.figure_format", ["pdf", "png", "svg"])
    validate_enum(config, "evaluation.table_format", ["csv", "parquet"])
    if "agent" in config:
        validate_enum(config, "agent.name", ["dqn", "doubledqn"])

    validate_pairs_have_metadata(config)
    validate_cost_order(config)
    if "agent" in config:
        validate_training_consistency(config)

    raw_data_dir = Path(_get_value(config, "data.raw_data_dir"))
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Configured raw data directory does not exist: {raw_data_dir}")

    return config


def resolve_config(
    root: Union[str, Path],
    agent_config: Optional[str] = None,
    experiment_overrides: Optional[List[str]] = None,
    benchmark_overrides: Optional[List[str]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a fully resolved configuration by merging in the canonical order.

    Parameters
    ----------
    root : path
        Repository root directory.
    agent_config : str, optional
        Relative path to agent YAML (e.g. ``configs/agents/dqn.yaml``).
    experiment_overrides : list of str, optional
        Paths to experiment/variant override YAMLs.
    benchmark_overrides : list of str, optional
        Paths to benchmark override YAMLs.
    cli_overrides : dict, optional
        Final command-line overrides applied last.
    """
    root = Path(root)
    merged: Dict[str, Any] = {}

    # 1-9: base merge order
    for rel in _BASE_MERGE_ORDER:
        cfg_path = _resolve_config_path(root, rel, "Base config file")
        merged = deep_merge(merged, load_yaml(cfg_path))

    # 10: agent config
    if agent_config:
        agent_path = _resolve_config_path(root, agent_config, "Agent config file")
        merged = deep_merge(merged, load_yaml(agent_path))

    # 11: experiment overrides
    if experiment_overrides:
        for rel_path in experiment_overrides:
            override_path = _resolve_config_path(root, rel_path, "Experiment override")
            merged = deep_merge(merged, load_yaml(override_path))

    # 12: benchmark overrides
    if benchmark_overrides:
        for rel_path in benchmark_overrides:
            override_path = _resolve_config_path(root, rel_path, "Benchmark override")
            merged = deep_merge(merged, load_yaml(override_path))

    # 13: CLI overrides
    if cli_overrides:
        merged = deep_merge(merged, cli_overrides)

    # Add resolution metadata
    merged.setdefault("meta", {})
    merged["meta"]["resolved_at"] = datetime.now(timezone.utc).isoformat()

    return validate_config(merged, root)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_required_keys(config: Dict[str, Any], required: List[str]) -> None:
    """Raise ``KeyError`` if any dot-delimited key is missing from *config*."""
    for key_path in required:
        parts = key_path.split(".")
        node = config
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"Missing required config key: {key_path}")
            node = node[part]


def validate_positive(config: Dict[str, Any], keys: List[str]) -> None:
    """Raise ``ValueError`` if any value at *keys* is not positive."""
    for key_path in keys:
        node = _get_value(config, key_path)
        if node is not None and node <= 0:
            raise ValueError(f"Config key {key_path} must be positive, got {node}")


def validate_range(config: Dict[str, Any], key: str, lo: float, hi: float) -> None:
    """Raise ``ValueError`` if the value at *key* is outside [lo, hi]."""
    node = _get_value(config, key)
    if node is not None and not (lo <= node <= hi):
        raise ValueError(f"Config key {key} must be in [{lo}, {hi}], got {node}")


def validate_enum(config: Dict[str, Any], key: str, allowed: List[Any]) -> None:
    """Raise ``ValueError`` if the value at *key* is not in *allowed*."""
    node = _get_value(config, key)
    if node not in allowed:
        raise ValueError(f"Config key {key} must be one of {allowed}, got {node}")


def normalize_paths(config: Dict[str, Any], root: Union[str, Path], keys: List[str]) -> Dict[str, Any]:
    """Resolve relative paths in *config* against *root* for the given *keys*."""
    root = Path(root)
    config = copy.deepcopy(config)
    for key_path in keys:
        parts = key_path.split(".")
        node = config
        for part in parts[:-1]:
            if part in node:
                node = node[part]
            else:
                break
        else:
            last = parts[-1]
            if last in node and isinstance(node[last], str):
                p = Path(node[last])
                if not p.is_absolute():
                    node[last] = str(root / p)
    return config


def save_resolved_config(config: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Save the resolved configuration to *output_path*."""
    return save_yaml(config, output_path)
