"""Portfolio manager — account-level state transitions and PnL application."""

from typing import Any, Dict
from src.environment.state.portfolio_state import PortfolioState
from src.environment.portfolio.position_manager import _pip_value


class PortfolioManager:
    """Manages account-level updates: cash, equity, margin, drawdown, turnover."""

    def __init__(self, config: Dict[str, Any]):
        env_cfg = config.get("environment", config)
        leverage_cfg = env_cfg.get("leverage", {})
        self.initial_margin_ratio = leverage_cfg.get("initial_margin_ratio", 0.0333333333)
        self.maintenance_margin_ratio = leverage_cfg.get("maintenance_margin_ratio", 0.5)
        self.initial_capital = env_cfg.get("account", {}).get("initial_capital", 100000.0)

    def apply_realized_pnl(self, state: PortfolioState, pnl: float) -> PortfolioState:
        """Add realized PnL to cash."""
        state.cash += pnl
        state.realized_pnl += pnl
        return state

    def update_unrealized_pnl(self, state: PortfolioState, mark_price: float, pip_size: float, pair_metadata: Dict) -> PortfolioState:
        """Recompute unrealized PnL from current mark price."""
        if state.direction == 0 or state.total_lots == 0:
            state.unrealized_pnl = 0.0
        else:
            price_diff = mark_price - state.average_entry_price
            pip_val = _pip_value(pip_size, mark_price, pair_metadata)
            state.unrealized_pnl = state.direction * price_diff / pip_size * pip_val * state.total_lots
        return state

    def update_margin(self, state: PortfolioState, mark_price: float, pip_size: float, pair_metadata: Dict) -> PortfolioState:
        """Update used margin based on current position."""
        if state.total_lots == 0:
            state.used_margin = 0.0
        else:
            lot_units = 100000
            notional = state.total_lots * lot_units * mark_price
            # For USDJPY, convert to USD
            quote_currency = pair_metadata.get("quote_currency", "USD")
            if quote_currency != "USD" and mark_price > 0:
                notional = state.total_lots * lot_units  # already in USD terms for USDJPY base
            state.used_margin = notional * self.initial_margin_ratio
        state.update_equity()
        return state

    def update_turnover(
        self,
        state: PortfolioState,
        traded_lots: float,
        price: float,
        pair_metadata: Dict[str, Any] = None,
    ) -> PortfolioState:
        """Add to cumulative turnover in account currency (USD)."""
        quote_currency = (pair_metadata or {}).get("quote_currency", "USD")
        notional = abs(traded_lots) * 100000 * price
        if quote_currency != "USD" and price > 0:
            notional /= price
        state.turnover += notional
        return state

    def apply_costs(self, state: PortfolioState, total_cost: float) -> PortfolioState:
        """Deduct execution costs from cash.

        Calls update_equity with update_peak=False because unrealized_pnl has not yet been
        marked to the current bar close — the peak should only advance after the final
        update_margin call at end-of-step.
        """
        state.cash -= total_cost
        state.update_equity(update_peak=False)
        return state
