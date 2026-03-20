"""Progress-bar utilities with optional nesting and logging-safe output.

This module centralizes tqdm setup so loops can stay concise while preserving
consistent formatting, colors, and configurability across training/workflows.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, Optional

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

_DEFAULT_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
_VERBOSITY_ORDER = {"quiet": 0, "normal": 1, "verbose": 2}
_DEFAULT_COLOURS = {
    "training": "cyan",
    "episode": "green",
    "experiment": "magenta",
    "evaluation": "blue",
    "benchmark": "yellow",
    "pair": "white",
    "agent": "red",
    "workflow": "bright_blue",
}


@contextmanager
def progress_logging_redirect(config: Optional[Dict[str, Any]], level: str = "normal"):
    """Redirect logger output through tqdm so logs don't break bar rendering."""
    if should_show_progress(config, level=level):
        with logging_redirect_tqdm():
            yield
        return
    yield


def should_show_progress(config: Optional[Dict[str, Any]], level: str = "normal") -> bool:
    """Return whether progress bars should render for the requested verbosity level."""
    opts = _progress_options(config)
    req = _VERBOSITY_ORDER.get(str(level or "normal").lower(), _VERBOSITY_ORDER["normal"])
    cur = _VERBOSITY_ORDER.get(opts["verbosity"], _VERBOSITY_ORDER["normal"])
    return bool(opts["enabled"] and cur >= req)


def get_progress_colour(config: Optional[Dict[str, Any]], key: str, fallback: Optional[str] = None) -> Optional[str]:
    """Resolve a colour token for tqdm bars from config or module defaults."""
    opts = _progress_options(config)
    colours = opts.get("colours", {})
    return colours.get(key, fallback or _DEFAULT_COLOURS.get(key))


def progress_iter(
    iterable: Iterable[Any],
    config: Optional[Dict[str, Any]],
    *,
    desc: str,
    total: Optional[int] = None,
    position: int = 0,
    leave: Optional[bool] = None,
    unit: str = "it",
    colour: Optional[str] = None,
    level: str = "normal",
):
    """Wrap an iterable with tqdm using project defaults and config overrides."""
    opts = _progress_options(config)
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        disable=not should_show_progress(config, level=level),
        position=opts["position_offset"] + max(0, int(position)),
        leave=_resolve_leave(opts, position, leave),
        unit=unit,
        dynamic_ncols=opts["dynamic_ncols"],
        mininterval=opts["refresh_rate"],
        miniters=opts["miniters"],
        ascii=opts["ascii"],
        bar_format=opts["bar_format"],
        colour=colour,
    )


def progress_bar(
    config: Optional[Dict[str, Any]],
    *,
    desc: str,
    total: Optional[int],
    initial: int = 0,
    position: int = 0,
    leave: Optional[bool] = None,
    unit: str = "it",
    colour: Optional[str] = None,
    level: str = "normal",
):
    """Create a tqdm bar for while-loop style progress tracking."""
    opts = _progress_options(config)
    return tqdm(
        total=total,
        initial=initial,
        desc=desc,
        disable=not should_show_progress(config, level=level),
        position=opts["position_offset"] + max(0, int(position)),
        leave=_resolve_leave(opts, position, leave),
        unit=unit,
        dynamic_ncols=opts["dynamic_ncols"],
        mininterval=opts["refresh_rate"],
        miniters=opts["miniters"],
        ascii=opts["ascii"],
        bar_format=opts["bar_format"],
        colour=colour,
    )


def set_bar_metrics(
    pbar: Optional[tqdm],
    *,
    metrics: Optional[Dict[str, Any]] = None,
    refresh: bool = False,
) -> None:
    """Set dynamic inline metrics on a tqdm bar, if available and enabled."""
    if pbar is None or metrics is None:
        return
    if not metrics:
        return
    pbar.set_postfix(metrics, refresh=refresh)


def with_progress_offset(config: Dict[str, Any], offset_delta: int) -> Dict[str, Any]:
    """Return a deep-copied config with progress bar position offset adjusted."""
    cfg = copy.deepcopy(config)
    logging_cfg = cfg.setdefault("logging", {})
    progress_cfg = logging_cfg.setdefault("progress", {})
    current = int(progress_cfg.get("position_offset", 0) or 0)
    progress_cfg["position_offset"] = current + int(offset_delta)
    return cfg


def _resolve_leave(opts: Dict[str, Any], position: int, leave: Optional[bool]) -> bool:
    if leave is not None:
        return bool(leave)
    if int(position) <= 0:
        return bool(opts["leave_outer"])
    return bool(opts["leave_inner"])


def _progress_options(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    logging_cfg = (config or {}).get("logging", {})
    progress_cfg = logging_cfg.get("progress", {})

    verbosity = str(progress_cfg.get("verbosity", logging_cfg.get("evaluation_verbosity", "normal"))).lower()
    if verbosity not in _VERBOSITY_ORDER:
        verbosity = "normal"

    refresh_rate = float(progress_cfg.get("refresh_rate", 0.1))
    if refresh_rate <= 0:
        refresh_rate = 0.1

    miniters = int(progress_cfg.get("miniters", 1))
    if miniters <= 0:
        miniters = 1

    return {
        "enabled": bool(progress_cfg.get("enabled", True)),
        "verbosity": verbosity,
        "refresh_rate": refresh_rate,
        "miniters": miniters,
        "dynamic_ncols": bool(progress_cfg.get("dynamic_ncols", True)),
        "ascii": bool(progress_cfg.get("ascii", False)),
        "leave_outer": bool(progress_cfg.get("leave_outer", True)),
        "leave_inner": bool(progress_cfg.get("leave_inner", False)),
        "bar_format": str(progress_cfg.get("bar_format", _DEFAULT_BAR_FORMAT)),
        "position_offset": int(progress_cfg.get("position_offset", 0) or 0),
        "colours": progress_cfg.get("colours", {}) or {},
    }
