"""Canonical action space enumeration.

The integer mapping is permanently locked and must never be reordered.
"""

from enum import IntEnum
from typing import Dict, List


class Action(IntEnum):
    """Extended 10-action enumeration."""
    HOLD = 0
    OPEN_LONG = 1
    OPEN_SHORT = 2
    PYRAMID_LONG = 3
    PYRAMID_SHORT = 4
    MARTINGALE_LONG = 5
    MARTINGALE_SHORT = 6
    REDUCE_POSITION = 7
    CLOSE_POSITION = 8
    REVERSE_POSITION = 9


class SimplifiedAction(IntEnum):
    """Simplified 3-action adapter."""
    HOLD = 0
    TARGET_LONG = 1
    TARGET_SHORT = 2


NUM_EXTENDED_ACTIONS = len(Action)
NUM_SIMPLIFIED_ACTIONS = len(SimplifiedAction)

ACTION_NAMES: Dict[int, str] = {a.value: a.name for a in Action}
SIMPLIFIED_ACTION_NAMES: Dict[int, str] = {a.value: a.name for a in SimplifiedAction}


def translate_simplified_action(action: int, current_direction: int) -> int:
    """Convert a simplified action to the extended action given current direction.

    Parameters
    ----------
    action : int
        Simplified action (0=HOLD, 1=TARGET_LONG, 2=TARGET_SHORT).
    current_direction : int
        -1 = short, 0 = flat, 1 = long.

    Returns
    -------
    int
        Extended action integer.
    """
    if action == SimplifiedAction.HOLD:
        return Action.HOLD
    elif action == SimplifiedAction.TARGET_LONG:
        if current_direction == 0:
            return Action.OPEN_LONG
        elif current_direction == -1:
            return Action.REVERSE_POSITION
        else:
            return Action.HOLD  # already long
    elif action == SimplifiedAction.TARGET_SHORT:
        if current_direction == 0:
            return Action.OPEN_SHORT
        elif current_direction == 1:
            return Action.REVERSE_POSITION
        else:
            return Action.HOLD  # already short
    return Action.HOLD
