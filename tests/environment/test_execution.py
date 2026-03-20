"""Tests for execution model — fill timing, cost ordering, reversal accounting."""

import numpy as np
import pytest

from src.environment.state.market_state import MarketState
from src.environment.market.execution_model import execute_trade, ExecutionResult
from src.environment.market.slippage_model import compute_slippage
from src.environment.market.transaction_cost_model import compute_transaction_costs
from src.environment.market.rollover_model import compute_rollover_cost
from src.environment.market.session_model import label_session
import pandas as pd


PAIR_META_USD = {"quote_currency": "USD"}


def _market(
    open_: float,
    close_: float,
    pip_size: float = 0.0001,
    ts=None,
    spread_proxy: float = 0.0,
    volatility_proxy: float = 0.0,
) -> MarketState:
    return MarketState(
        timestamp=ts or pd.Timestamp("2023-06-01 10:00:00"),
        pair="EURUSD",
        open=open_,
        high=max(open_, close_) + 0.005,
        low=min(open_, close_) - 0.005,
        close=close_,
        volume=1000.0,
        spread_proxy=spread_proxy,
        volatility_proxy=volatility_proxy,
        pip_size=pip_size,
    )


def _config() -> dict:
    return {
        "environment": {
            "slippage": {
                "enabled": True,
                "mode": "deterministic",
                "base_slippage_pips": 0.5,
                "volatility_multiplier": 1.0,
                "session_multiplier": 1.0,
            },
            "transaction_costs": {
                "enabled": True,
                "spread_mode": "fixed",
                "fixed_spread_pips": 1.0,
                "commission_per_lot": 3.5,
                "cost_multiplier": 1.0,
            },
            "rollover": {
                "enabled": False,
            },
        }
    }


class TestFillTiming:
    """Fill price must be based on next bar open, not current bar."""

    def test_buy_fills_at_next_open_plus_slippage(self):
        current = _market(1.1000, 1.1050)
        next_bar = _market(1.1060, 1.1080)  # open=1.1060
        result = execute_trade(1, 0.1, current, next_bar, PAIR_META_USD, _config())
        # Fill should be next_bar.open + slippage (0.5 pips = 0.00005)
        assert result.base_reference_price == 1.1060
        assert result.fill_price > result.base_reference_price  # slippage adverse on buy

    def test_sell_fills_at_next_open_minus_slippage(self):
        current = _market(1.1000, 1.1050)
        next_bar = _market(1.1060, 1.1080)
        result = execute_trade(-1, 0.1, current, next_bar, PAIR_META_USD, _config())
        assert result.fill_price < result.base_reference_price

    def test_fill_never_uses_current_close(self):
        """Anti-lookahead: fill must use next_bar.open, not current.close."""
        current = _market(1.10, 1.12)
        next_bar = _market(1.11, 1.13)
        result = execute_trade(1, 0.1, current, next_bar, PAIR_META_USD, _config())
        assert abs(result.base_reference_price - 1.11) < 1e-10
        assert abs(result.base_reference_price - 1.12) > 0.001  # not current close


class TestCostOrdering:
    """Spread, slippage, commission costs must all be present and positive."""

    def test_all_cost_components_present(self):
        current = _market(1.10, 1.1050)
        next_bar = _market(1.1060, 1.1080)
        result = execute_trade(1, 0.1, current, next_bar, PAIR_META_USD, _config())
        assert result.spread_cost >= 0
        assert result.commission_cost >= 0
        assert result.slippage_cost >= 0
        assert result.total_cost > 0

    def test_disabled_slippage_zero(self):
        cfg = _config()
        cfg["environment"]["slippage"]["enabled"] = False
        current = _market(1.10, 1.1050)
        next_bar = _market(1.1060, 1.1080)
        result = execute_trade(1, 0.1, current, next_bar, PAIR_META_USD, cfg)
        assert result.slippage_cost == 0.0
        assert result.fill_price == result.base_reference_price

    def test_cost_proxies_use_decision_bar_by_default(self):
        cfg = _config()
        cfg["environment"]["transaction_costs"]["spread_mode"] = "proxy"

        current = _market(1.10, 1.1050, spread_proxy=0.0001)
        next_bar = _market(1.1060, 1.1080, spread_proxy=0.0050)
        result = execute_trade(1, 0.1, current, next_bar, PAIR_META_USD, cfg)

        expected = compute_transaction_costs(
            lots=0.1,
            fill_price=result.fill_price,
            pip_size=next_bar.pip_size,
            spread_proxy=current.spread_proxy,
            pair_metadata=PAIR_META_USD,
            config=cfg["environment"]["transaction_costs"],
        )
        assert result.spread_cost == pytest.approx(expected["spread_cost"], rel=1e-8)

    def test_cost_proxies_can_use_next_bar_when_enabled(self):
        cfg = _config()
        cfg["environment"]["transaction_costs"]["spread_mode"] = "proxy"
        cfg["environment"]["execution"] = {"use_next_bar_market_proxies": True}

        current = _market(1.10, 1.1050, spread_proxy=0.0001)
        next_bar = _market(1.1060, 1.1080, spread_proxy=0.0050)
        result = execute_trade(1, 0.1, current, next_bar, PAIR_META_USD, cfg)

        expected = compute_transaction_costs(
            lots=0.1,
            fill_price=result.fill_price,
            pip_size=next_bar.pip_size,
            spread_proxy=next_bar.spread_proxy,
            pair_metadata=PAIR_META_USD,
            config=cfg["environment"]["transaction_costs"],
        )
        assert result.spread_cost == pytest.approx(expected["spread_cost"], rel=1e-8)


class TestReversalAccounting:
    """Reversal is close+open: both legs must have costs."""

    def test_reversal_costs_both_legs(self):
        # Simulated via two separate execute_trade calls (as done in trading_env)
        current = _market(1.10, 1.1050)
        next_bar = _market(1.1060, 1.1080)
        close_result = execute_trade(-1, 0.1, current, next_bar, PAIR_META_USD, _config())  # close long
        open_result = execute_trade(-1, 0.1, current, next_bar, PAIR_META_USD, _config())   # open short
        total = close_result.total_cost + open_result.total_cost
        assert total > close_result.total_cost  # both legs contribute


class TestCostMultiplier:
    """Transaction cost robustness multipliers."""

    def test_2x_multiplier(self):
        cfg = _config()
        current = _market(1.10, 1.1050)
        next_bar = _market(1.1060, 1.1080)

        result_1x = execute_trade(1, 0.1, current, next_bar, PAIR_META_USD, cfg)
        base_spread = result_1x.spread_cost
        base_comm = result_1x.commission_cost

        cfg["environment"]["transaction_costs"]["cost_multiplier"] = 2.0
        result_2x = execute_trade(1, 0.1, current, next_bar, PAIR_META_USD, cfg)
        assert result_2x.spread_cost == pytest.approx(base_spread * 2.0, rel=0.01)
        assert result_2x.commission_cost == pytest.approx(base_comm * 2.0, rel=0.01)


class TestSessionModel:
    def test_session_labels(self):
        assert label_session(3) == "Asia"
        assert label_session(10) == "London"
        assert label_session(15) == "Overlap"
        assert label_session(19) == "NewYork"
        assert label_session(23) == "OffHours"


class TestRollover:
    def test_no_rollover_when_flat(self):
        cost = compute_rollover_cost(
            pd.Timestamp("2023-06-07 22:00:00"), 0, 0.1, {"enabled": True, "cutoff_hour_utc": 22, "annual_rate_long": 0.02}
        )
        assert cost == 0.0

    def test_rollover_on_cutoff_hour(self):
        cost = compute_rollover_cost(
            pd.Timestamp("2023-06-07 22:00:00"), 1, 0.1,
            {"enabled": True, "cutoff_hour_utc": 22, "annual_rate_long": 0.02, "annual_rate_short": -0.015, "triple_rollover_day": "Wednesday"}
        )
        # Wednesday = triple rollover
        assert cost != 0.0

    def test_no_rollover_wrong_hour(self):
        cost = compute_rollover_cost(
            pd.Timestamp("2023-06-07 10:00:00"), 1, 0.1,
            {"enabled": True, "cutoff_hour_utc": 22, "annual_rate_long": 0.02}
        )
        assert cost == 0.0
