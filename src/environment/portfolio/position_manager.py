"""Position manager — handles position sizing, entry prices, and transitions."""

from typing import Any, Dict, Tuple
from src.environment.state.portfolio_state import PortfolioState


class PositionManager:
    """Manages position opens, adds (pyramid/martingale), reduces, closes, and reversals."""

    def __init__(self, config: Dict[str, Any]):
        env_cfg = config.get("environment", config)
        actions_cfg = env_cfg.get("actions", {})
        self.base_lot_size = actions_cfg.get("base_lot_size", 0.1)
        self.reduce_fraction = actions_cfg.get("reduce_fraction", 0.5)
        self.pyramid_increment = actions_cfg.get("pyramid_increment_lots", 0.1)
        self.martingale_multiplier = actions_cfg.get("martingale_multiplier", 2.0)

    def open_position(self, state: PortfolioState, direction: int, fill_price: float) -> PortfolioState:
        """Open a new position from flat."""
        state.direction = direction
        state.total_lots = self.base_lot_size
        state.average_entry_price = fill_price
        state.pyramid_levels = 0
        state.martingale_steps = 0
        state.trade_count += 1
        return state

    def add_pyramid(self, state: PortfolioState, fill_price: float) -> PortfolioState:
        """Add to position via pyramid (winning direction)."""
        add_lots = self.pyramid_increment
        new_total = state.total_lots + add_lots
        # Weighted average entry price
        state.average_entry_price = (
            state.average_entry_price * state.total_lots + fill_price * add_lots
        ) / new_total
        state.total_lots = new_total
        state.pyramid_levels += 1
        state.trade_count += 1
        return state

    def add_martingale(self, state: PortfolioState, fill_price: float) -> PortfolioState:
        """Add to position via martingale (adverse direction)."""
        add_lots = state.total_lots * (self.martingale_multiplier - 1.0)
        if add_lots < self.base_lot_size:
            add_lots = self.base_lot_size
        new_total = state.total_lots + add_lots
        state.average_entry_price = (
            state.average_entry_price * state.total_lots + fill_price * add_lots
        ) / new_total
        state.total_lots = new_total
        state.martingale_steps += 1
        state.trade_count += 1
        return state

    def reduce_position(self, state: PortfolioState, fill_price: float, pip_size: float, pair_metadata: Dict) -> Tuple[PortfolioState, float]:
        """Reduce position by configured fraction. Returns (state, realized_pnl)."""
        reduce_lots = state.total_lots * self.reduce_fraction
        if reduce_lots < self.base_lot_size:
            reduce_lots = state.total_lots  # close entirely
        remaining = state.total_lots - reduce_lots

        pnl = self._compute_pnl(state.direction, state.average_entry_price, fill_price, reduce_lots, pip_size, pair_metadata)

        if remaining < 1e-10:
            state = self._flatten(state)
        else:
            state.total_lots = remaining
            # Entry price unchanged on reduction

        state.trade_count += 1
        return state, pnl

    def close_position(self, state: PortfolioState, fill_price: float, pip_size: float, pair_metadata: Dict) -> Tuple[PortfolioState, float]:
        """Close entire position. Returns (state, realized_pnl)."""
        pnl = self._compute_pnl(state.direction, state.average_entry_price, fill_price, state.total_lots, pip_size, pair_metadata)
        state = self._flatten(state)
        state.trade_count += 1
        return state, pnl

    def reverse_position(self, state: PortfolioState, fill_price: float, pip_size: float, pair_metadata: Dict) -> Tuple[PortfolioState, float]:
        """Close current position and open opposite. Returns (state, realized_pnl)."""
        # Close leg
        pnl = self._compute_pnl(state.direction, state.average_entry_price, fill_price, state.total_lots, pip_size, pair_metadata)
        new_direction = -state.direction
        state = self._flatten(state)
        # Open leg
        state = self.open_position(state, new_direction, fill_price)
        return state, pnl

    @staticmethod
    def _compute_pnl(direction: int, entry_price: float, exit_price: float, lots: float, pip_size: float, pair_metadata: Dict) -> float:
        """Compute PnL in account currency."""
        price_diff = exit_price - entry_price
        pip_value_per_lot = _pip_value(pip_size, exit_price, pair_metadata)
        pnl = direction * price_diff / pip_size * pip_value_per_lot * lots
        return pnl

    @staticmethod
    def _flatten(state: PortfolioState) -> PortfolioState:
        state.direction = 0
        state.total_lots = 0.0
        state.average_entry_price = 0.0
        state.pyramid_levels = 0
        state.martingale_steps = 0
        return state


def _pip_value(pip_size: float, current_price: float, pair_metadata: Dict) -> float:
    """Return pip value in account currency (USD) for one standard lot (100k units).

    For XXX/USD pairs the pip value is pip_size * 100000.
    For USD/JPY the pip value is (pip_size / current_price) * 100000.
    """
    quote_currency = pair_metadata.get("quote_currency", "USD")
    lot_units = 100000  # standard lot
    if quote_currency == "USD":
        return pip_size * lot_units
    else:
        # e.g. USDJPY: convert JPY pips to USD
        if current_price > 0:
            return (pip_size / current_price) * lot_units
        return pip_size * lot_units
