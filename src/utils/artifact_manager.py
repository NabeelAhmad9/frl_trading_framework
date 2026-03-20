"""Canonical artifact path manager.

All output directory creation must go through this module. No workflow, script,
or notebook should hardcode artifact paths.
"""

from pathlib import Path
from typing import Optional, Union


class ArtifactManager:
    """Create and resolve canonical artifact directory paths.

    Parameters
    ----------
    outputs_root : path
        Root directory for all generated artifacts (default ``outputs/``).
    """

    def __init__(self, outputs_root: Union[str, Path] = "outputs"):
        self.root = Path(outputs_root).resolve()

    # ------------------------------------------------------------------
    # Experiment artifacts
    # ------------------------------------------------------------------

    def experiment_variant_dir(self, experiment: str, variant: str) -> Path:
        """Return and create the directory for one experiment variant."""
        d = self.root / "experiments" / experiment / variant
        d.mkdir(parents=True, exist_ok=True)
        return d

    def experiment_summary_dir(self, experiment: str) -> Path:
        d = self.root / "experiments" / experiment / "summary"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Final result artifacts (agents)
    # ------------------------------------------------------------------

    def agent_result_dir(self, agent: str, pair: str) -> Path:
        d = self.root / "results" / "agents" / agent / pair
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Benchmark artifacts
    # ------------------------------------------------------------------

    def benchmark_result_dir(self, benchmark: str, pair: str) -> Path:
        d = self.root / "results" / "benchmarks" / benchmark / pair
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Comparison artifacts
    # ------------------------------------------------------------------

    def comparison_all_pairs_dir(self, comparison_name: str = "dqn_vs_doubledqn") -> Path:
        d = self.root / "results" / "comparisons" / comparison_name / "all_pairs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def comparison_by_pair_dir(self, pair: str, comparison_name: str = "dqn_vs_doubledqn") -> Path:
        d = self.root / "results" / "comparisons" / comparison_name / "by_pair" / pair
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Sub-directory helpers (common structure)
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_subdirs(base: Path, subdirs: Optional[list] = None) -> dict:
        """Create canonical sub-directories under *base* and return a mapping."""
        base = Path(base)
        base.mkdir(parents=True, exist_ok=True)
        if subdirs is None:
            subdirs = [
                "checkpoints",
                "models",
                "metrics/train",
                "metrics/train/episodes",
                "metrics/test",
                "figures/train",
                "figures/test",
                "tables/train",
                "tables/test",
                "logs",
            ]
        result = {}
        for sd in subdirs:
            p = base / sd
            p.mkdir(parents=True, exist_ok=True)
            result[sd.replace("/", "_")] = p
        return result

    def setup_run(self, base: Path) -> dict:
        """Create all canonical sub-directories for a training/eval run."""
        return self.ensure_subdirs(base)

    # ------------------------------------------------------------------
    # Convenience: resolved config path
    # ------------------------------------------------------------------

    @staticmethod
    def resolved_config_path(base: Path) -> Path:
        return base / "resolved_config.yaml"
