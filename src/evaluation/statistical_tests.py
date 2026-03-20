"""Statistical tests — conservative comparison with not_applicable fallback."""

import numpy as np
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def paired_bootstrap_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap test on return series.

    Parameters
    ----------
    returns_a, returns_b : array
        Per-period return arrays from two agents on the same data.
    n_bootstrap : int
        Number of bootstrap samples.
    seed : int

    Returns
    -------
    dict with keys: statistic, p_value, significant, method
    """
    a = np.asarray(returns_a, dtype=np.float64)
    b = np.asarray(returns_b, dtype=np.float64)

    if len(a) < 30 or len(b) < 30:
        return {
            "statistic": None,
            "p_value": None,
            "significant": None,
            "method": "not_applicable",
            "reason": "insufficient_samples",
        }

    diff = a[:min(len(a), len(b))] - b[:min(len(a), len(b))]
    observed = float(np.mean(diff))
    n = len(diff)

    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_bootstrap):
        sample = diff[rng.randint(0, n, size=n)]
        if np.mean(sample) >= observed:
            count += 1

    p_value = count / n_bootstrap

    return {
        "statistic": observed,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "method": "paired_bootstrap",
    }


def compare_agents(
    agent_returns: Dict[str, np.ndarray],
    method: str = "bootstrap_or_not_applicable",
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare multiple agents pairwise.

    Parameters
    ----------
    agent_returns : dict
        agent_name → return array.
    method : str
    seed : int

    Returns
    -------
    dict of pairwise comparison results.
    """
    names = sorted(agent_returns.keys())
    results = {}

    if len(names) < 2:
        return {"result": "not_applicable", "reason": "only_one_agent"}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}_vs_{names[j]}"
            if method == "bootstrap_or_not_applicable":
                results[key] = paired_bootstrap_test(
                    agent_returns[names[i]],
                    agent_returns[names[j]],
                    seed=seed,
                )
            else:
                results[key] = {
                    "method": "not_applicable",
                    "reason": f"unsupported_method_{method}",
                }

    return results
