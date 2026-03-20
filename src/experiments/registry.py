"""Experiment registry — discover experiment families and metadata."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config_loader import load_yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)


def discover_experiments(experiments_root: Path) -> List[Dict[str, Any]]:
    """Scan experiments directory and return metadata for each family.

    Parameters
    ----------
    experiments_root : Path
        e.g. ``configs/experiments/``.

    Returns
    -------
    list of dict
        Each dict has keys ``name``, ``path``, ``metadata``, ``variants``.
    """
    experiments_root = Path(experiments_root)
    families = []

    for d in sorted(experiments_root.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "experiment.yaml"
        if not meta_path.exists():
            continue

        meta = load_yaml(meta_path).get("experiment", {})
        variant_dir = d / meta.get("variant_dir", "variants")
        variants = []
        if variant_dir.is_dir():
            for vf in sorted(variant_dir.glob("*.yaml")):
                variants.append({"name": vf.stem, "path": vf})

        families.append({
            "name": meta.get("name", d.name),
            "path": d,
            "metadata": meta,
            "variants": variants,
        })

    logger.info("Discovered %d experiment families", len(families))
    return families


def get_experiment(experiments_root: Path, name: str) -> Optional[Dict[str, Any]]:
    """Get a single experiment family by name."""
    for exp in discover_experiments(experiments_root):
        if exp["name"] == name:
            return exp
    return None
