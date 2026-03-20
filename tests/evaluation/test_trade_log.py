"""Tests for canonical trade-event extraction."""

from src.evaluation.trade_log import build_trade_event


class TestTradeLogExtraction:
    def test_non_trade_step_returns_none(self):
        info = {
            "direction_before": 1,
            "direction": 1,
            "realized_pnl": 0.0,
            "forced_liquidation": False,
            "pyramid_levels": 1,
            "martingale_steps": 0,
        }
        assert build_trade_event(info, step=10) is None

    def test_trade_event_uses_before_levels_on_flatten(self):
        info = {
            "direction_before": 1,
            "direction": 0,
            "realized_pnl": 25.0,
            "forced_liquidation": False,
            "pyramid_levels_before": 2,
            "pyramid_levels": 0,
            "martingale_steps_before": 3,
            "martingale_steps": 0,
            "turnover_delta": 15000.0,
            "total_lots_before": 0.4,
            "total_lots_after": 0.0,
            "equity_after": 101000.0,
        }
        event = build_trade_event(info, step=11)
        assert event is not None
        assert event["pyramid_steps"] == 2
        assert event["martingale_steps"] == 3
        assert event["notional"] == 15000.0

    def test_trade_event_falls_back_to_lot_equity_notional(self):
        info = {
            "direction_before": -1,
            "direction": 0,
            "realized_pnl": -10.0,
            "forced_liquidation": False,
            "pyramid_levels_before": 0,
            "pyramid_levels": 0,
            "martingale_steps_before": 1,
            "martingale_steps": 0,
            "turnover_delta": 0.0,
            "total_lots_before": 0.2,
            "total_lots_after": 0.0,
            "equity_after": 100000.0,
        }
        event = build_trade_event(info, step=5)
        assert event is not None
        assert event["notional"] == 20000.0
