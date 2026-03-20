"""Risk constraints — margin, leverage, and liquidation rules."""

from typing import Any, Dict
from src.environment.state.portfolio_state import PortfolioState


class RiskConstraints:
    """Centralized risk checks for margin, leverage, exposure, and liquidation."""

    def __init__(self, config: Dict[str, Any]):
        env_cfg = config.get("environment", config)
        actions_cfg = env_cfg.get("actions", {})
        leverage_cfg = env_cfg.get("leverage", {})
        liq_cfg = env_cfg.get("liquidation", {})

        self.max_total_position = actions_cfg.get("max_total_position_size", 2.0)
        self.max_lots_per_dir = actions_cfg.get("max_open_lots_per_direction", 2.0)
        self.max_pyramid_levels = actions_cfg.get("max_pyramid_levels", 2)
        self.max_martingale_steps = actions_cfg.get("max_martingale_steps", 2)
        self.profit_threshold = actions_cfg.get("profit_threshold", 0.001)
        self.adverse_threshold = actions_cfg.get("adverse_threshold", 0.001)
        self.initial_margin_ratio = leverage_cfg.get("initial_margin_ratio", 0.0333333333)
        self.liquidation_threshold = liq_cfg.get("threshold", 0.25)
        self.liquidation_enabled = liq_cfg.get("enabled", True)
        self.base_lot_size = actions_cfg.get("base_lot_size", 0.1)
        self.pyramid_increment = actions_cfg.get("pyramid_increment_lots", 0.1)
        self.martingale_multiplier = actions_cfg.get("martingale_multiplier", 2.0)

    def can_open_position(self, state: PortfolioState, lot_size: float, price: float, pair: str = None) -> bool:
        """Check if there is enough margin to open or add a position."""
        if lot_size <= 0 or price <= 0:
            return False

        if not self.check_total_exposure(state, lot_size):
            return False

        if state.direction != 0 and (state.total_lots + lot_size) > self.max_lots_per_dir:
            return False

        pair_norm = (pair or "").replace("/", "").upper()
        if pair_norm.startswith("USD"):
            # USD base pairs (e.g. USDJPY): notional is already in account currency.
            notional = lot_size * 100000
        else:
            # Non-USD base pairs (e.g. EURUSD): convert notional via current price.
            notional = lot_size * 100000 * price

        required_margin = notional * self.initial_margin_ratio
        return state.free_margin >= required_margin

    def check_total_exposure(self, state: PortfolioState, add_lots: float) -> bool:
        """Check if adding lots would exceed max total position size."""
        return (state.total_lots + add_lots) <= self.max_total_position

    def can_pyramid(self, state: PortfolioState, current_price: float) -> bool:
        """Check pyramid conditions: profit threshold, level cap, exposure cap."""
        if state.direction == 0:
            return False
        if state.pyramid_levels >= self.max_pyramid_levels:
            return False
        if not self.check_total_exposure(state, self.pyramid_increment):
            return False
        # Profit threshold check
        if state.average_entry_price > 0:
            pct_change = (current_price - state.average_entry_price) / state.average_entry_price
            if state.direction == 1 and pct_change < self.profit_threshold:
                return False
            if state.direction == -1 and (-pct_change) < self.profit_threshold:
                return False
        return True

    def can_martingale(self, state: PortfolioState, current_price: float) -> bool:
        """Check martingale conditions: adverse threshold, step cap, exposure cap."""
        if state.direction == 0:
            return False
        if state.martingale_steps >= self.max_martingale_steps:
            return False
        add_lots = max(state.total_lots * (self.martingale_multiplier - 1.0), self.base_lot_size)
        if not self.check_total_exposure(state, add_lots):
            return False
        # Adverse movement check
        if state.average_entry_price > 0:
            pct_change = (current_price - state.average_entry_price) / state.average_entry_price
            if state.direction == 1 and pct_change > -self.adverse_threshold:
                return False
            if state.direction == -1 and (-pct_change) > -self.adverse_threshold:
                return False
        return True

    def check_liquidation(self, state: PortfolioState) -> bool:
        """Return True if liquidation threshold is breached."""
        if not self.liquidation_enabled:
            return False
        if state.used_margin <= 0:
            return False
        margin_level = state.equity / state.used_margin if state.used_margin > 0 else float("inf")
        return margin_level < self.liquidation_threshold

    def margin_utilization(self, state: PortfolioState) -> float:
        """Return margin utilization ratio."""
        if state.equity <= 0:
            return float("inf")
        return state.used_margin / state.equity
