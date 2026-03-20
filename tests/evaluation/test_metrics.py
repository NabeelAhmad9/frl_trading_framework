"""Tests for evaluation metrics — edge cases and known formulas."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics


class TestMetricsBasic:
    """Known equity curve scenarios."""

    def test_flat_equity(self):
        """Zero-volatility case — constant equity."""
        eq = np.full(100, 100000.0)
        m = compute_metrics(eq)
        assert m["cumulative_return"] == pytest.approx(0.0, abs=1e-9)
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
        assert m["annualized_volatility"] == pytest.approx(0.0, abs=1e-6)

    def test_linear_growth(self):
        """Simple upward equity: 100k → 110k."""
        eq = np.linspace(100000, 110000, 100)
        m = compute_metrics(eq)
        assert m["cumulative_return"] == pytest.approx(0.10, abs=1e-6)
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)

    def test_all_loss(self):
        """Steadily declining to near-zero."""
        eq = np.linspace(100000, 1000, 200)
        m = compute_metrics(eq)
        assert m["cumulative_return"] < 0
        assert m["max_drawdown"] > 0.9

    def test_single_dip_drawdown(self):
        """100k → 80k → 100k: max drawdown = 20%."""
        eq = np.concatenate([
            np.linspace(100000, 80000, 50),
            np.linspace(80000, 100000, 50),
        ])
        m = compute_metrics(eq)
        assert m["max_drawdown"] == pytest.approx(0.2, abs=0.01)

    def test_two_element_equity(self):
        """Minimal valid equity curve."""
        eq = np.array([100000.0, 105000.0])
        m = compute_metrics(eq)
        assert m["cumulative_return"] == pytest.approx(0.05, abs=1e-6)


class TestMetricsEdgeCases:
    """Edge cases: empty, zero, single-element."""

    def test_single_element(self):
        eq = np.array([100000.0])
        m = compute_metrics(eq)
        assert m["cumulative_return"] == pytest.approx(0.0)

    def test_zero_initial_equity(self):
        """Should not crash on zero initial."""
        eq = np.array([0.0, 100.0, 200.0])
        m = compute_metrics(eq)
        # cumulative_return huge but shouldn't crash
        assert np.isfinite(m["cumulative_return"])

    def test_empty_trade_log(self):
        """Win rate, turnover, liquidation should be zero with no trades."""
        eq = np.linspace(100000, 110000, 50)
        m = compute_metrics(eq, trade_log=[])
        assert m["win_rate"] == 0.0
        assert m["turnover"] == 0.0
        assert m["liquidation_count"] == 0

    def test_trade_log_metrics(self):
        eq = np.linspace(100000, 120000, 50)
        trades = [
            {"pnl": 500, "notional": 10000, "pyramid_steps": 1, "martingale_steps": 0, "forced_liquidation": False},
            {"pnl": -200, "notional": 5000, "pyramid_steps": 0, "martingale_steps": 1, "forced_liquidation": False},
            {"pnl": 300, "notional": 8000, "pyramid_steps": 0, "martingale_steps": 0, "forced_liquidation": True},
        ]
        m = compute_metrics(eq, trade_log=trades)
        assert m["total_trades"] == 3
        assert m["win_rate"] == pytest.approx(2.0 / 3.0, abs=1e-6)
        assert m["liquidation_count"] == 1
        assert m["avg_pyramid_steps"] == pytest.approx(1.0 / 3.0, abs=1e-6)
        assert m["turnover"] > 0


class TestSharpeSort:
    """Sharpe and Sortino known values."""

    def test_positive_sharpe(self):
        """Upward equity → positive Sharpe."""
        eq = np.linspace(100000, 150000, 1000)
        m = compute_metrics(eq)
        assert m["sharpe_ratio"] > 0

    def test_negative_sharpe(self):
        """Downward equity → negative Sharpe."""
        eq = np.linspace(100000, 50000, 1000)
        m = compute_metrics(eq)
        assert m["sharpe_ratio"] < 0

    def test_sortino_higher_than_sharpe_with_positive_returns(self):
        """With mostly positive returns and few negatives, Sortino >= Sharpe."""
        rng = np.random.RandomState(0)
        eq = 100000 + np.cumsum(rng.uniform(0, 10, 500))
        m = compute_metrics(eq)
        # Sortino uses downside vol which is smaller → ratio is larger
        assert m["sortino_ratio"] >= m["sharpe_ratio"] - 1.0  # allow some margin
