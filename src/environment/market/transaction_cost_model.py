"""Transaction cost model — spread and commission computation."""

from typing import Any, Dict


def compute_transaction_costs(
    lots: float,
    fill_price: float,
    pip_size: float,
    spread_proxy: float,
    pair_metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Compute spread and commission costs in account currency (USD).

    Parameters
    ----------
    lots : float
        Number of lots traded.
    fill_price : float
        Execution price.
    pip_size : float
    spread_proxy : float
        Market spread proxy in price terms.
    pair_metadata : dict
        Must contain 'quote_currency'.
    config : dict
        transaction_costs config block.

    Returns
    -------
    dict
        Keys: spread_cost, commission_cost, total_cost (all in account currency).
    """
    if not config.get("enabled", True):
        return {"spread_cost": 0.0, "commission_cost": 0.0, "total_cost": 0.0}

    cost_multiplier = config.get("cost_multiplier", 1.0)
    lot_units = 100000
    quote_currency = pair_metadata.get("quote_currency", "USD")

    # Spread cost
    spread_mode = config.get("spread_mode", "proxy")
    if spread_mode == "proxy" and spread_proxy > 0:
        spread_price = spread_proxy
    else:
        spread_price = config.get("fixed_spread_pips", 1.0) * pip_size

    spread_cost_raw = spread_price * lots * lot_units
    if quote_currency != "USD" and fill_price > 0:
        spread_cost_raw /= fill_price
    spread_cost = spread_cost_raw * cost_multiplier

    # Commission cost
    commission_per_lot = config.get("commission_per_lot", 3.5)
    commission_cost = commission_per_lot * lots * cost_multiplier

    return {
        "spread_cost": spread_cost,
        "commission_cost": commission_cost,
        "total_cost": spread_cost + commission_cost,
    }
