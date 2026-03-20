"""Agent registry — build agents from resolved config."""

from typing import Any, Dict

from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

_AGENT_REGISTRY: Dict[str, type] = {}


def register_agent(name: str):
    """Decorator to register an agent class under a canonical name."""
    def decorator(cls):
        _AGENT_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def build_agent(
    input_dim: int,
    num_actions: int,
    config: Dict[str, Any],
    device: str = "cpu",
) -> BaseAgent:
    """Instantiate the correct agent class from resolved config.

    The agent type is resolved from ``config.agent.name`` with
    ``double_dqn`` flag as a secondary discriminator.

    Parameters
    ----------
    input_dim : int
        Flattened observation dimension.
    num_actions : int
        Size of the discrete action space.
    config : dict
        Fully resolved configuration.
    device : str
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    BaseAgent
        Instantiated DQN or DoubleDQN agent.
    """
    # Lazy imports to avoid circular dependencies at module load time.
    from src.agents.dqn.dqn_agent import DQNAgent
    from src.agents.doubledqn.doubledqn_agent import DoubleDQNAgent

    agent_cfg = config.get("agent", {})
    agent_name = agent_cfg.get("name", "dqn").lower()
    is_double = agent_cfg.get("algorithm", {}).get("double_dqn", False)

    # Resolve class: explicit name wins; fallback to double_dqn flag.
    if agent_name == "doubledqn" or agent_name == "double_dqn" or is_double:
        cls = DoubleDQNAgent
        resolved_name = "doubledqn"
    else:
        cls = DQNAgent
        resolved_name = "dqn"

    logger.info("Building agent: %s (input_dim=%d, num_actions=%d)", resolved_name, input_dim, num_actions)
    return cls(input_dim, num_actions, config, device=device)


def list_agents() -> list:
    """Return names of all available agent types."""
    return ["dqn", "doubledqn"]
