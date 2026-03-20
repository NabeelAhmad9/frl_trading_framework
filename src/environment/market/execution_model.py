"""Execution model — combine timing, fill price, slippage, spread, commission, rollover."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from src.environment.market.slippage_model import compute_slippage
from src.environment.market.transaction_cost_model import compute_transaction_costs
from src.environment.market.rollover_model import compute_rollover_cost
from src.environment.market.session_model import label_session, session_multiplier
from src.environment.state.market_state import MarketState


@dataclass
class ExecutionResult:
    """Execution output for one trade leg."""
    fill_price: float
    base_reference_price: float
    spread_cost: float
    commission_cost: float
    slippage_cost: float
    rollover_cost: float
    total_cost: float
    execution_timestamp: object


def execute_trade(
    trade_direction: int,
    lots: float,
    current_market: MarketState,
    next_market: MarketState,
    pair_metadata: Dict[str, Any],
    config: Dict[str, Any],
    portfolio_direction: int = 0,
    portfolio_lots: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
) -> ExecutionResult:
    """Execute a trade with anti-lookahead timing.

    Fill at next_market.open, cost accounting per config order.

    Parameters
    ----------
    trade_direction : int
        1=buy, -1=sell.
    lots : float
        Lots to trade.
    current_market : MarketState
        Bar at decision time t.
    next_market : MarketState
        Bar at execution time t+1.
    pair_metadata : dict
        Contains 'quote_currency'.
    config : dict
        Full environment config (or just the relevant blocks).
    portfolio_direction : int
        Current portfolio direction for rollover.
    portfolio_lots : float
        Current portfolio lots for rollover.
    rng : RandomState or None
        For stochastic slippage.

    Returns
    -------
    ExecutionResult
    """
    env_cfg = config.get("environment", config)
    execution_cfg = env_cfg.get("execution", {})
    slippage_cfg = env_cfg.get("slippage", {})
    tc_cfg = env_cfg.get("transaction_costs", {})
    rollover_cfg = env_cfg.get("rollover", {})

    pip_size = next_market.pip_size
    base_price = next_market.open

    # Session multiplier
    sess_label = next_market.session_label
    sess_mult = session_multiplier(sess_label, env_cfg)

    # Anti-lookahead: spread/volatility proxies are commonly derived from full-bar statistics
    # (e.g. high-low ranges). Those are not observable at open_{t+1}. Use decision-time (t)
    # proxies by default unless explicitly overridden.
    use_next_bar_proxies = execution_cfg.get("use_next_bar_market_proxies", False)
    proxy_market = next_market if use_next_bar_proxies else current_market

    # Slippage (always adverse to agent)
    slippage_amount = compute_slippage(
        direction=trade_direction,
        pip_size=pip_size,
        config=slippage_cfg,
        volatility_proxy=proxy_market.volatility_proxy,
        session_mult=sess_mult,
        rng=rng,
    )

    # Fill price: buy higher, sell lower
    if trade_direction == 1:
        fill_price = base_price + slippage_amount
    else:
        fill_price = base_price - slippage_amount

    # Transaction costs
    tc = compute_transaction_costs(
        lots=lots,
        fill_price=fill_price,
        pip_size=pip_size,
        spread_proxy=proxy_market.spread_proxy,
        pair_metadata=pair_metadata,
        config=tc_cfg,
    )

    # Rollover cost
    rollover = compute_rollover_cost(
        timestamp=next_market.timestamp,
        direction=portfolio_direction,
        total_lots=portfolio_lots,
        config=rollover_cfg,
    )

    # FIX: Respect rollover sign — negative rollover means a credit to the agent (carry-positive
    # short). The previous abs() inverted that credit into an extra cost.
    # FIX: Slippage is already embedded in fill_price (buy higher / sell lower), so it reduces
    # realized PnL when the trade is eventually closed. Adding slippage_cost_usd on top as a
    # separate cash deduction was counting the same cost twice. Keep it as a reporting field only.
    total = tc["total_cost"] + rollover   # spread + commission + rollover (sign-correct)

    slippage_cost_usd = slippage_amount * lots * 100000
    quote_currency = pair_metadata.get("quote_currency", "USD")
    if quote_currency != "USD" and fill_price > 0:
        slippage_cost_usd /= fill_price

    return ExecutionResult(
        fill_price=fill_price,
        base_reference_price=base_price,
        spread_cost=tc["spread_cost"],
        commission_cost=tc["commission_cost"],
        slippage_cost=slippage_cost_usd,   # reporting only — already priced into fill_price
        rollover_cost=rollover,
        total_cost=total,
        execution_timestamp=next_market.timestamp,
    )
