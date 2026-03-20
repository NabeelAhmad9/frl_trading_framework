"""Tests for portfolio management, position transitions, risk constraints, and legal masks."""

import numpy as np
import pytest

from src.environment.action_space import Action
from src.environment.state.market_state import MarketState
from src.environment.state.portfolio_state import PortfolioState
from src.environment.portfolio.position_manager import PositionManager
from src.environment.portfolio.portfolio_manager import PortfolioManager
from src.environment.portfolio.risk_constraints import RiskConstraints
from src.environment.legal_action_mask import compute_legal_mask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
PAIR_META_USD = {"quote_currency": "USD"}
PAIR_META_JPY = {"quote_currency": "JPY"}
PIP_SIZE_USD = 0.0001
PIP_SIZE_JPY = 0.01


def _make_config(**overrides):
    """Build a minimal valid config dict."""
    cfg = {
        "environment": {
            "actions": {
                "base_lot_size": 0.1,
                "reduce_fraction": 0.5,
                "pyramid_increment_lots": 0.1,
                "martingale_multiplier": 2.0,
                "max_total_position_size": 2.0,
                "max_open_lots_per_direction": 2.0,
                "max_pyramid_levels": 2,
                "max_martingale_steps": 2,
                "profit_threshold": 0.001,
                "adverse_threshold": 0.001,
            },
            "leverage": {
                "initial_margin_ratio": 0.0333333333,
                "maintenance_margin_ratio": 0.5,
            },
            "liquidation": {
                "threshold": 0.25,
            },
            "account": {
                "initial_capital": 100000.0,
            },
        }
    }
    for key, val in overrides.items():
        cfg["environment"]["actions"][key] = val
    return cfg


def _make_market(close: float = 1.10, pip_size: float = PIP_SIZE_USD) -> MarketState:
    return MarketState(
        timestamp=None,
        pair="EURUSD",
        open=close - 0.001,
        high=close + 0.005,
        low=close - 0.005,
        close=close,
        volume=1000.0,
        pip_size=pip_size,
    )


# ---------------------------------------------------------------------------
# PortfolioState tests
# ---------------------------------------------------------------------------
class TestPortfolioState:

    def test_initial_factory(self):
        ps = PortfolioState.initial(50000.0)
        assert ps.cash == 50000.0
        assert ps.equity == 50000.0
        assert ps.direction == 0
        assert ps.total_lots == 0.0

    def test_copy_independence(self):
        ps = PortfolioState.initial()
        cp = ps.copy()
        cp.cash = 0.0
        assert ps.cash == 100000.0

    def test_update_equity_drawdown(self):
        ps = PortfolioState.initial()
        ps.unrealized_pnl = -5000.0
        ps.update_equity()
        assert ps.equity == 95000.0
        assert ps.current_drawdown == pytest.approx(0.05)

    def test_peak_equity_tracking(self):
        ps = PortfolioState.initial()
        ps.unrealized_pnl = 10000.0
        ps.update_equity()
        assert ps.peak_equity == 110000.0
        ps.unrealized_pnl = 5000.0
        ps.update_equity()
        assert ps.peak_equity == 110000.0
        assert ps.current_drawdown > 0


# ---------------------------------------------------------------------------
# PositionManager tests
# ---------------------------------------------------------------------------
class TestPositionManager:

    def setup_method(self):
        self.config = _make_config()
        self.pm = PositionManager(self.config)

    def test_open_long(self):
        ps = PortfolioState.initial()
        ps = self.pm.open_position(ps, 1, 1.10)
        assert ps.direction == 1
        assert ps.total_lots == 0.1
        assert ps.average_entry_price == 1.10
        assert ps.trade_count == 1

    def test_open_short(self):
        ps = PortfolioState.initial()
        ps = self.pm.open_position(ps, -1, 1.10)
        assert ps.direction == -1

    def test_pyramid_weighted_avg(self):
        ps = PortfolioState.initial()
        ps = self.pm.open_position(ps, 1, 1.10)
        ps = self.pm.add_pyramid(ps, 1.12)
        assert ps.total_lots == pytest.approx(0.2)
        expected_avg = (1.10 * 0.1 + 1.12 * 0.1) / 0.2
        assert ps.average_entry_price == pytest.approx(expected_avg)
        assert ps.pyramid_levels == 1

    def test_martingale_doubles(self):
        ps = PortfolioState.initial()
        ps = self.pm.open_position(ps, 1, 1.10)
        ps = self.pm.add_martingale(ps, 1.08)
        assert ps.total_lots == pytest.approx(0.2)
        assert ps.martingale_steps == 1

    def test_reduce_halves(self):
        ps = PortfolioState.initial()
        ps = self.pm.open_position(ps, 1, 1.10)
        ps.total_lots = 0.4
        ps, pnl = self.pm.reduce_position(ps, 1.12, PIP_SIZE_USD, PAIR_META_USD)
        assert ps.total_lots == pytest.approx(0.2)
        assert pnl > 0  # price went up on a long

    def test_close_flattens(self):
        ps = PortfolioState.initial()
        ps = self.pm.open_position(ps, 1, 1.10)
        ps, pnl = self.pm.close_position(ps, 1.12, PIP_SIZE_USD, PAIR_META_USD)
        assert ps.direction == 0
        assert ps.total_lots == 0.0
        assert pnl > 0

    def test_reverse_flips_direction(self):
        ps = PortfolioState.initial()
        ps = self.pm.open_position(ps, 1, 1.10)
        ps, pnl = self.pm.reverse_position(ps, 1.12, PIP_SIZE_USD, PAIR_META_USD)
        assert ps.direction == -1
        assert ps.total_lots == 0.1
        assert pnl > 0

    def test_reduce_small_converts_to_close(self):
        """When reduce fraction leaves < base lot, close entirely."""
        ps = PortfolioState.initial()
        ps = self.pm.open_position(ps, 1, 1.10)
        # total_lots=0.1, reduce fraction=0.5 → reduce_lots=0.05 < base(0.1) → close
        ps, pnl = self.pm.reduce_position(ps, 1.12, PIP_SIZE_USD, PAIR_META_USD)
        assert ps.direction == 0
        assert ps.total_lots == 0.0


# ---------------------------------------------------------------------------
# PortfolioManager tests
# ---------------------------------------------------------------------------
class TestPortfolioManager:

    def setup_method(self):
        self.config = _make_config()
        self.pfm = PortfolioManager(self.config)

    def test_apply_realized_pnl(self):
        ps = PortfolioState.initial()
        ps = self.pfm.apply_realized_pnl(ps, 500.0)
        assert ps.cash == 100500.0
        assert ps.realized_pnl == 500.0

    def test_update_unrealized_flat(self):
        ps = PortfolioState.initial()
        ps = self.pfm.update_unrealized_pnl(ps, 1.10, PIP_SIZE_USD, PAIR_META_USD)
        assert ps.unrealized_pnl == 0.0

    def test_update_unrealized_long_profit(self):
        ps = PortfolioState.initial()
        ps.direction = 1
        ps.total_lots = 0.1
        ps.average_entry_price = 1.10
        ps = self.pfm.update_unrealized_pnl(ps, 1.12, PIP_SIZE_USD, PAIR_META_USD)
        # +200 pips * $10/pip * 0.1 lots = $200
        assert ps.unrealized_pnl == pytest.approx(200.0)

    def test_apply_costs_reduces_cash(self):
        ps = PortfolioState.initial()
        ps = self.pfm.apply_costs(ps, 50.0)
        assert ps.cash == 99950.0

    def test_turnover_usdjpy_converted_to_usd(self):
        ps = PortfolioState.initial()
        ps = self.pfm.update_turnover(ps, traded_lots=1.0, price=150.0, pair_metadata=PAIR_META_JPY)
        assert ps.turnover == pytest.approx(100000.0)


# ---------------------------------------------------------------------------
# RiskConstraints tests
# ---------------------------------------------------------------------------
class TestRiskConstraints:

    def setup_method(self):
        self.config = _make_config()
        self.rc = RiskConstraints(self.config)

    def test_can_open_with_margin(self):
        ps = PortfolioState.initial()
        assert self.rc.can_open_position(ps, 0.1, 1.10)

    def test_cannot_open_without_margin(self):
        ps = PortfolioState.initial()
        ps.free_margin = 0.0
        assert not self.rc.can_open_position(ps, 0.1, 1.10)

    def test_can_open_usdjpy_with_reasonable_margin(self):
        ps = PortfolioState.initial()
        ps.free_margin = 10000.0
        assert self.rc.can_open_position(ps, lot_size=1.0, price=150.0, pair="USDJPY")

    def test_exposure_cap(self):
        ps = PortfolioState.initial()
        ps.total_lots = 1.9
        assert self.rc.check_total_exposure(ps, 0.1)
        assert not self.rc.check_total_exposure(ps, 0.2)

    def test_pyramid_cap(self):
        ps = PortfolioState.initial()
        ps.direction = 1
        ps.total_lots = 0.1
        ps.average_entry_price = 1.10
        ps.pyramid_levels = 2  # at max
        assert not self.rc.can_pyramid(ps, 1.12)

    def test_martingale_cap(self):
        ps = PortfolioState.initial()
        ps.direction = 1
        ps.total_lots = 0.1
        ps.average_entry_price = 1.10
        ps.martingale_steps = 2  # at max
        assert not self.rc.can_martingale(ps, 1.08)

    def test_liquidation_trigger(self):
        ps = PortfolioState.initial()
        ps.equity = 100.0
        ps.used_margin = 1000.0
        assert self.rc.check_liquidation(ps)  # margin_level = 0.1 < 0.25

    def test_no_liquidation_when_flat(self):
        ps = PortfolioState.initial()
        assert not self.rc.check_liquidation(ps)


# ---------------------------------------------------------------------------
# Legal action mask tests
# ---------------------------------------------------------------------------
class TestLegalMask:

    def setup_method(self):
        self.config = _make_config()
        self.rc = RiskConstraints(self.config)
        self.market = _make_market()

    def test_flat_allows_hold_open_only(self):
        ps = PortfolioState.initial()
        mask = compute_legal_mask(ps, self.market, self.rc)
        assert mask[Action.HOLD] == 1.0
        assert mask[Action.OPEN_LONG] == 1.0
        assert mask[Action.OPEN_SHORT] == 1.0
        assert mask[Action.CLOSE_POSITION] == 0.0
        assert mask[Action.REDUCE_POSITION] == 0.0
        assert mask[Action.REVERSE_POSITION] == 0.0
        assert mask[Action.PYRAMID_LONG] == 0.0
        assert mask[Action.PYRAMID_SHORT] == 0.0
        assert mask[Action.MARTINGALE_LONG] == 0.0
        assert mask[Action.MARTINGALE_SHORT] == 0.0

    def test_long_allows_close_reduce_reverse(self):
        ps = PortfolioState.initial()
        ps.direction = 1
        ps.total_lots = 0.1
        ps.average_entry_price = 1.10
        mask = compute_legal_mask(ps, self.market, self.rc)
        assert mask[Action.HOLD] == 1.0
        assert mask[Action.CLOSE_POSITION] == 1.0
        assert mask[Action.REDUCE_POSITION] == 1.0
        assert mask[Action.REVERSE_POSITION] == 1.0
        assert mask[Action.OPEN_LONG] == 0.0
        assert mask[Action.OPEN_SHORT] == 0.0

    def test_no_illegal_action_admitted(self):
        """Exhaust random portfolio/market combos to ensure no impossible actions leak."""
        rng = np.random.RandomState(42)
        for _ in range(200):
            ps = PortfolioState.initial()
            ps.direction = rng.choice([-1, 0, 1])
            ps.total_lots = float(rng.uniform(0, 2.5))
            ps.average_entry_price = float(rng.uniform(0.5, 2.0)) if ps.direction != 0 else 0.0
            ps.pyramid_levels = int(rng.randint(0, 4))
            ps.martingale_steps = int(rng.randint(0, 4))
            ps.cash = float(rng.uniform(0, 200000))
            ps.equity = ps.cash
            ps.used_margin = float(rng.uniform(0, 50000))
            ps.free_margin = ps.equity - ps.used_margin
            ps.peak_equity = max(ps.equity, 100000)
            market = _make_market(close=float(rng.uniform(0.8, 1.5)))
            mask = compute_legal_mask(ps, market, self.rc)
            # HOLD always legal
            assert mask[Action.HOLD] == 1.0
            # If flat, position-specific actions must be 0
            if ps.direction == 0:
                assert mask[Action.CLOSE_POSITION] == 0.0
                assert mask[Action.REDUCE_POSITION] == 0.0
                assert mask[Action.REVERSE_POSITION] == 0.0
                assert mask[Action.PYRAMID_LONG] == 0.0
                assert mask[Action.PYRAMID_SHORT] == 0.0
                assert mask[Action.MARTINGALE_LONG] == 0.0
                assert mask[Action.MARTINGALE_SHORT] == 0.0
            # If non-flat, cannot open new position
            if ps.direction != 0:
                assert mask[Action.OPEN_LONG] == 0.0
                assert mask[Action.OPEN_SHORT] == 0.0

    def test_simplified_mask_always_full(self):
        ps = PortfolioState.initial()
        mask = compute_legal_mask(ps, self.market, self.rc, simplified_mode=True)
        assert mask.shape == (3,)
        assert np.all(mask == 1.0)

    def test_mask_dtype_float32(self):
        ps = PortfolioState.initial()
        mask = compute_legal_mask(ps, self.market, self.rc)
        assert mask.dtype == np.float32
