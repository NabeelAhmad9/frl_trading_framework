"""Variant builder — resolve per-variant configs from base + overrides."""

from pathlib import Path
from typing import Any, Dict, List

from src.utils.config_loader import deep_merge, load_yaml, resolve_config, save_resolved_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_variant(
    root: Path,
    experiment_meta: Dict[str, Any],
    variant_path: Path,
) -> Dict[str, Any]:
    """Deep-merge base configs with variant overrides.

    Parameters
    ----------
    root : Path
        Repository root.
    experiment_meta : dict
        Experiment metadata from ``experiment.yaml``.
    variant_path : Path
        Path to variant YAML file.

    Returns
    -------
    dict
        Fully resolved config for this variant.
    """
    # Start with full canonical merge using the experiment's base agent
    agent_name = experiment_meta.get("base_agent", "doubledqn")
    agent_yaml = f"configs/agents/{agent_name}.yaml"

    config = resolve_config(root=str(root), agent_config=agent_yaml)

    # Add experiment metadata
    config["experiment"] = {
        "name": experiment_meta.get("name", ""),
        "description": experiment_meta.get("description", ""),
        "base_pair": experiment_meta.get("base_pair", "EURUSD"),
        "base_agent": agent_name,
        "variant": variant_path.stem,
    }

    # Deep-merge variant overrides
    variant_overrides = load_yaml(variant_path)
    config = deep_merge(config, variant_overrides)

    logger.info("Resolved variant: %s/%s", experiment_meta.get("name", ""), variant_path.stem)
    return config


def resolve_all_variants(
    root: Path,
    experiment: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Resolve configs for all variants in an experiment family.

    Parameters
    ----------
    root : Path
    experiment : dict
        From ``registry.discover_experiments``.

    Returns
    -------
    list of dict
        Each dict has ``name``, ``config``, ``variant_path``.
    """
    results = []
    meta = experiment["metadata"]

    for v in experiment["variants"]:
        config = resolve_variant(root, meta, v["path"])
        results.append({
            "name": v["name"],
            "config": config,
            "variant_path": v["path"],
        })

    return results
