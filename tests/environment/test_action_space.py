"""Tests for action space and simplified mode translation."""

import pytest

from src.environment.action_space import (
    Action,
    SimplifiedAction,
    NUM_EXTENDED_ACTIONS,
    NUM_SIMPLIFIED_ACTIONS,
    ACTION_NAMES,
    SIMPLIFIED_ACTION_NAMES,
    translate_simplified_action,
)


class TestActionEnum:
    """Action enumeration must be permanently frozen."""

    def test_extended_count(self):
        assert NUM_EXTENDED_ACTIONS == 10

    def test_simplified_count(self):
        assert NUM_SIMPLIFIED_ACTIONS == 3

    def test_canonical_ordering(self):
        assert Action.HOLD == 0
        assert Action.OPEN_LONG == 1
        assert Action.OPEN_SHORT == 2
        assert Action.PYRAMID_LONG == 3
        assert Action.PYRAMID_SHORT == 4
        assert Action.MARTINGALE_LONG == 5
        assert Action.MARTINGALE_SHORT == 6
        assert Action.REDUCE_POSITION == 7
        assert Action.CLOSE_POSITION == 8
        assert Action.REVERSE_POSITION == 9

    def test_simplified_ordering(self):
        assert SimplifiedAction.HOLD == 0
        assert SimplifiedAction.TARGET_LONG == 1
        assert SimplifiedAction.TARGET_SHORT == 2

    def test_action_names_complete(self):
        assert len(ACTION_NAMES) == 10
        assert ACTION_NAMES[0] == "HOLD"
        assert ACTION_NAMES[9] == "REVERSE_POSITION"

    def test_simplified_names_complete(self):
        assert len(SIMPLIFIED_ACTION_NAMES) == 3


class TestSimplifiedTranslation:
    """Simplified 3-action adapter must map correctly given position state."""

    def test_hold_from_flat(self):
        assert translate_simplified_action(0, 0) == Action.HOLD

    def test_target_long_from_flat(self):
        assert translate_simplified_action(1, 0) == Action.OPEN_LONG

    def test_target_short_from_flat(self):
        assert translate_simplified_action(2, 0) == Action.OPEN_SHORT

    def test_target_long_from_short_reverses(self):
        assert translate_simplified_action(1, -1) == Action.REVERSE_POSITION

    def test_target_short_from_long_reverses(self):
        assert translate_simplified_action(2, 1) == Action.REVERSE_POSITION

    def test_target_long_while_long_holds(self):
        assert translate_simplified_action(1, 1) == Action.HOLD

    def test_target_short_while_short_holds(self):
        assert translate_simplified_action(2, -1) == Action.HOLD
