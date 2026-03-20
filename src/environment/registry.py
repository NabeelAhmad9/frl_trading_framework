"""Environment registry — build full or simplified environments from config."""

from typing import Any, Dict
import pandas as pd

from src.environment.trading_env import TradingEnv


def make_env(
    df: pd.DataFrame,
    pair: str,
    config: Dict[str, Any],
    reward_engine: Any = None,
    simplified: bool = None,
) -> TradingEnv:
    """Build a TradingEnv from resolved config.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-ready dataset.
    pair : str
    config : dict
        Resolved config.
    reward_engine : Any
        Optional reward engine.
    simplified : bool or None
        If provided, overrides config.environment.actions.simplified_mode.

    Returns
    -------
    TradingEnv
    """
    if simplified is not None:
        config = _override_simplified(config, simplified)
    return TradingEnv(df=df, pair=pair, config=config, reward_engine=reward_engine)


def _override_simplified(config: Dict[str, Any], simplified: bool) -> Dict[str, Any]:
    import copy
    cfg = copy.deepcopy(config)
    env = cfg.setdefault("environment", {})
    actions = env.setdefault("actions", {})
    actions["simplified_mode"] = simplified
    return cfg
