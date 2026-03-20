"""Portfolio state container — mutable account and position state."""

from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class PortfolioState:
    """Mutable portfolio and position state."""
    cash: float = 100000.0
    equity: float = 100000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    used_margin: float = 0.0
    free_margin: float = 100000.0
    maintenance_margin_ratio: float = 0.5
    direction: int = 0  # -1=short, 0=flat, 1=long
    total_lots: float = 0.0
    average_entry_price: float = 0.0
    pyramid_levels: int = 0
    martingale_steps: int = 0
    trade_count: int = 0
    turnover: float = 0.0
    peak_equity: float = 100000.0
    current_drawdown: float = 0.0
    last_action: int = 0
    forced_liquidations: int = 0

    def copy(self) -> "PortfolioState":
        # All fields are scalars/immutables, so dataclass replace is equivalent to
        # deepcopy here and significantly faster in tight environment loops.
        return replace(self)

    def update_equity(self, update_peak: bool = True) -> None:
        """Recompute equity and related fields.

        Parameters
        ----------
        update_peak : bool
            When False, the peak-equity high-water mark is NOT advanced. Pass False during
            intra-step intermediate calls (e.g. apply_costs) where unrealized_pnl has not yet
            been marked to the current bar. Pass True (default) only after the final mark-to-
            market is complete so the peak reflects a fully-settled equity figure.
        """
        self.equity = self.cash + self.unrealized_pnl
        if update_peak and self.equity > self.peak_equity:
            self.peak_equity = self.equity
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0
        self.free_margin = self.equity - self.used_margin

    @staticmethod
    def initial(initial_capital: float = 100000.0, maintenance_margin_ratio: float = 0.5) -> "PortfolioState":
        return PortfolioState(
            cash=initial_capital,
            equity=initial_capital,
            free_margin=initial_capital,
            peak_equity=initial_capital,
            maintenance_margin_ratio=maintenance_margin_ratio,
        )
