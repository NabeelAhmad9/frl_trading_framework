"""Gymnasium-compatible trading environment with anti-lookahead timing."""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from src.environment.action_space import (
    Action, NUM_EXTENDED_ACTIONS, NUM_SIMPLIFIED_ACTIONS,
    ACTION_NAMES, translate_simplified_action,
)
from src.environment.state.portfolio_state import PortfolioState
from src.environment.legal_action_mask import compute_legal_mask
from src.environment.observation_builder import build_observation
from src.environment.portfolio.position_manager import PositionManager
from src.environment.portfolio.portfolio_manager import PortfolioManager
from src.environment.portfolio.risk_constraints import RiskConstraints
from src.environment.market.market_simulator import MarketSimulator
from src.environment.market.execution_model import execute_trade
from src.environment.market.rollover_model import compute_rollover_cost
from src.environment.market.transaction_cost_model import compute_transaction_costs


class TradingEnv(gym.Env):
    """Forex trading environment.

    Timeline per step:
        observe close_t → decide at t → execute at open_{t+1} →
        apply costs → mark at close_{t+1} → compute reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        pair: str,
        config: Dict[str, Any],
        reward_engine: Any = None,
    ):
        super().__init__()
        if len(df) < 2:
            raise ValueError("TradingEnv requires at least 2 rows to support t -> t+1 execution timing.")

        self.config = config
        self.pair = pair
        env_cfg = config.get("environment", config)

        self.simplified_mode = env_cfg.get("actions", {}).get("simplified_mode", False)
        self.num_actions = NUM_SIMPLIFIED_ACTIONS if self.simplified_mode else NUM_EXTENDED_ACTIONS

        # Feature columns (exclude market columns and timestamp). Only numeric dtypes are used as features.
        market_cols = {"timestamp", "open", "high", "low", "close", "volume",
                       "Open", "High", "Low", "Close", "Volume"}
        self.feature_columns = [
            c for c in df.columns
            if c not in market_cols and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        if self.feature_columns:
            self._feature_matrix = df[self.feature_columns].values.astype(np.float32)
            self.num_features = len(self.feature_columns)
        else:
            # Keep the observation space valid even when no engineered features are present.
            self._feature_matrix = np.zeros((len(df), 1), dtype=np.float32)
            self.num_features = 1

        obs_cfg = env_cfg.get("observation", {})
        self.window_length = obs_cfg.get("window_length", 24)

        # Spaces
        portfolio_dim = 10
        flat_dim = self.window_length * self.num_features + portfolio_dim + self.num_actions
        self.observation_space = spaces.Dict({
            "market": spaces.Box(-np.inf, np.inf, shape=(self.window_length, self.num_features), dtype=np.float32),
            "portfolio": spaces.Box(-np.inf, np.inf, shape=(portfolio_dim,), dtype=np.float32),
            "mask": spaces.Box(0, 1, shape=(self.num_actions,), dtype=np.float32),
            "flat": spaces.Box(-np.inf, np.inf, shape=(flat_dim,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(self.num_actions)

        # Sub-components
        self.simulator = MarketSimulator(df, pair, config)
        self.position_mgr = PositionManager(config)
        self.portfolio_mgr = PortfolioManager(config)
        self.risk = RiskConstraints(config)
        self.reward_engine = reward_engine

        # Invalid action policy
        ia_cfg = env_cfg.get("invalid_action", {})
        self.invalid_action_policy = ia_cfg.get("policy", "convert_to_hold_with_penalty")
        self.invalid_action_penalty = ia_cfg.get("penalty", 0.0)

        # Episode config
        ep_cfg = env_cfg.get("episode", {})
        self.start_policy = ep_cfg.get("start_policy", "beginning")
        self.end_policy = ep_cfg.get("end_policy", "end_of_data")
        self.max_steps = ep_cfg.get("max_steps", None)

        # State
        self.portfolio: Optional[PortfolioState] = None
        self._step_count = 0
        self._rng: Optional[np.random.RandomState] = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        # FIX: Hardcoded seed=42 on every unseeded reset made stochastic slippage identical
        # across all training episodes. Now: explicit seed → deterministic; first call without
        # seed → truly random; subsequent calls without seed → reuse existing RNG (Gymnasium
        # convention — don't discard entropy unless the caller explicitly requests it).
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        elif self._rng is None:
            self._rng = np.random.RandomState()   # truly random seed on first reset

        env_cfg = self.config.get("environment", self.config)
        initial_capital = env_cfg.get("account", {}).get("initial_capital", 100000.0)
        self.portfolio = PortfolioState.initial(initial_capital)

        # Start cursor according to configured episode start policy.
        # Clamp to a valid decision index that still has a t+1 bar available.
        min_start_idx = min(max(self.window_length - 1, 0), max(self.simulator.length - 2, 0))
        max_start_idx = max(self.simulator.length - 2, 0)

        requested_start = None
        if isinstance(options, dict):
            requested_start = options.get("start_index")

        if requested_start is not None:
            try:
                start_idx = int(requested_start)
            except (TypeError, ValueError):
                start_idx = min_start_idx
            start_idx = int(min(max(start_idx, min_start_idx), max_start_idx))
        elif self.start_policy == "random" and max_start_idx >= min_start_idx:
            start_idx = int(self._rng.randint(min_start_idx, max_start_idx + 1))
        else:
            start_idx = min_start_idx

        self.simulator.reset(start_idx)
        self._step_count = 0

        # Reset stateful reward components (e.g., volatility/overtrading buffers)
        # so each episode starts with a clean reward state.
        if self.reward_engine is not None and hasattr(self.reward_engine, "reset_engine"):
            self.reward_engine.reset_engine()

        current_market = self.simulator.current_state()
        reset_mask = compute_legal_mask(self.portfolio, current_market, self.risk, self.simplified_mode)
        obs = self._build_obs(market_state=current_market, legal_mask=reset_mask)
        _ts = current_market.timestamp
        info = self._build_info(
            raw_action=0, executed_action=0, was_legal=True,
            cost_breakdown={}, reward_breakdown={},
            decision_ts=_ts, execution_ts=_ts, forced_liquidation=False,
            market_state=current_market,
            legal_mask=reset_mask,
        )
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        market_t = self.simulator.current_state()
        next_market = self.simulator.next_state()

        if next_market is None:
            # Episode over — no more bars
            terminal_mask = compute_legal_mask(self.portfolio, market_t, self.risk, self.simplified_mode)
            obs = self._build_obs(market_state=market_t, legal_mask=terminal_mask)
            _ts = market_t.timestamp
            info = self._build_info(
                raw_action=action, executed_action=Action.HOLD, was_legal=True,
                cost_breakdown={}, reward_breakdown={},
                decision_ts=_ts, execution_ts=_ts, forced_liquidation=False,
                market_state=market_t,
                legal_mask=terminal_mask,
            )
            return obs, 0.0, False, True, info

        # FIX: capture both timestamps before the cursor is advanced so the info dict can
        # correctly distinguish the decision bar (t) from the execution bar (t+1).
        decision_ts = market_t.timestamp
        execution_ts = next_market.timestamp

        # Validate incoming action before any mask indexing.
        raw_action = action
        invalid_action_input = False
        try:
            action_idx = int(action)
        except (TypeError, ValueError):
            action_idx = -1
            invalid_action_input = True

        if action_idx < 0 or action_idx >= self.num_actions:
            invalid_action_input = True

        if invalid_action_input:
            extended_action = int(Action.HOLD)
            was_legal = False
        else:
            # Translate simplified action.
            if self.simplified_mode:
                extended_action = int(translate_simplified_action(action_idx, self.portfolio.direction))
            else:
                extended_action = int(action_idx)

            # Compute legal mask and check legality.
            mask = compute_legal_mask(self.portfolio, market_t, self.risk, simplified_mode=False)
            was_legal = bool(mask[extended_action] == 1.0)

        # Invalid action handling
        actual_penalty = 0.0
        if not was_legal:
            if self.invalid_action_policy == "convert_to_hold_with_penalty":
                extended_action = Action.HOLD
                actual_penalty = self.invalid_action_penalty
            else:
                extended_action = Action.HOLD

        # Save pre-step state
        equity_before = self.portfolio.equity
        cash_before = self.portfolio.cash
        prev_portfolio = self.portfolio.copy()

        # Execute action
        cost_breakdown = {"spread_cost": 0.0, "commission_cost": 0.0, "slippage_cost": 0.0, "rollover_cost": 0.0, "total_cost": 0.0}
        realized_pnl = 0.0
        direction_before = self.portfolio.direction

        pair_meta = self.simulator.pair_metadata
        env_cfg = self.config.get("environment", self.config)
        rollover_cfg = env_cfg.get("rollover", {})
        tc_cfg = env_cfg.get("transaction_costs", {})

        if extended_action == Action.HOLD:
            # FIX: HOLD never applied rollover. Every open position must pay overnight financing
            # when the bar timestamp falls on the rollover cutoff hour, regardless of action.
            if self.portfolio.direction != 0:
                rollover = compute_rollover_cost(
                    timestamp=next_market.timestamp,
                    direction=self.portfolio.direction,
                    total_lots=self.portfolio.total_lots,
                    config=rollover_cfg,
                )
                if rollover != 0.0:
                    self.portfolio = self.portfolio_mgr.apply_costs(self.portfolio, rollover)
                    cost_breakdown["rollover_cost"] = rollover
                    cost_breakdown["total_cost"] = rollover

        elif extended_action == Action.OPEN_LONG:
            result = execute_trade(1, self.position_mgr.base_lot_size, market_t, next_market, pair_meta, self.config, self.portfolio.direction, self.portfolio.total_lots, self._rng)
            self.portfolio = self.position_mgr.open_position(self.portfolio, 1, result.fill_price)
            cost_breakdown = _result_to_costs(result)
            self.portfolio = self.portfolio_mgr.apply_costs(self.portfolio, result.total_cost)
            self.portfolio = self.portfolio_mgr.update_turnover(self.portfolio, self.position_mgr.base_lot_size, result.fill_price, pair_meta)

        elif extended_action == Action.OPEN_SHORT:
            result = execute_trade(-1, self.position_mgr.base_lot_size, market_t, next_market, pair_meta, self.config, self.portfolio.direction, self.portfolio.total_lots, self._rng)
            self.portfolio = self.position_mgr.open_position(self.portfolio, -1, result.fill_price)
            cost_breakdown = _result_to_costs(result)
            self.portfolio = self.portfolio_mgr.apply_costs(self.portfolio, result.total_cost)
            self.portfolio = self.portfolio_mgr.update_turnover(self.portfolio, self.position_mgr.base_lot_size, result.fill_price, pair_meta)

        elif extended_action == Action.PYRAMID_LONG:
            result = execute_trade(1, self.position_mgr.pyramid_increment, market_t, next_market, pair_meta, self.config, self.portfolio.direction, self.portfolio.total_lots, self._rng)
            self.portfolio = self.position_mgr.add_pyramid(self.portfolio, result.fill_price)
            cost_breakdown = _result_to_costs(result)
            self.portfolio = self.portfolio_mgr.apply_costs(self.portfolio, result.total_cost)
            self.portfolio = self.portfolio_mgr.update_turnover(self.portfolio, self.position_mgr.pyramid_increment, result.fill_price, pair_meta)

        elif extended_action == Action.PYRAMID_SHORT:
            result = execute_trade(-1, self.position_mgr.pyramid_increment, market_t, next_market, pair_meta, self.config, self.portfolio.direction, self.portfolio.total_lots, self._rng)
            self.portfolio = self.position_mgr.add_pyramid(self.portfolio, result.fill_price)
            cost_breakdown = _result_to_costs(result)
            self.portfolio = self.portfolio_mgr.apply_costs(self.portfolio, result.total_cost)
            self.portfolio = self.portfolio_mgr.update_turnover(self.portfolio, self.position_mgr.pyramid_increment, result.fill_price, pair_meta)

        elif extended_action == Action.MARTINGALE_LONG:
            add_lots = max(self.portfolio.total_lots * (self.position_mgr.martingale_multiplier - 1.0), self.position_mgr.base_lot_size)
            result = execute_trade(1, add_lots, market_t, next_market, pair_meta, self.config, self.portfolio.direction, self.portfolio.total_lots, self._rng)
            self.portfolio = self.position_mgr.add_martingale(self.portfolio, result.fill_price)
            cost_breakdown = _result_to_costs(result)
            self.portfolio = self.portfolio_mgr.apply_costs(self.portfolio, result.total_cost)
            self.portfolio = self.portfolio_mgr.update_turnover(self.portfolio, add_lots, result.fill_price, pair_meta)

        elif extended_action == Action.MARTINGALE_SHORT:
            add_lots = max(self.portfolio.total_lots * (self.position_mgr.martingale_multiplier - 1.0), self.position_mgr.base_lot_size)
            result = execute_trade(-1, add_lots, market_t, next_market, pair_meta, self.config, self.portfolio.direction, self.portfolio.total_lots, self._rng)
            self.portfolio = self.position_mgr.add_martingale(self.portfolio, result.fill_price)
            cost_breakdown = _result_to_costs(result)
            self.portfolio = self.portfolio_mgr.apply_costs(self.portfolio, result.total_cost)
            self.portfolio = self.portfolio_mgr.update_turnover(self.portfolio, add_lots, result.fill_price, pair_meta)

        elif extended_action == Action.REDUCE_POSITION:
            trade_dir = -self.portfolio.direction
            reduce_lots = self.portfolio.total_lots * self.position_mgr.reduce_fraction
            if reduce_lots < self.position_mgr.base_lot_size:
                reduce_lots = self.portfolio.total_lots
            result = execute_trade(trade_dir, reduce_lots, market_t, next_market, pair_meta, self.config, self.portfolio.direction, self.portfolio.total_lots, self._rng)
            self.portfolio, realized_pnl = self.position_mgr.reduce_position(self.portfolio, result.fill_price, self.simulator.pip_size, pair_meta)
            cost_breakdown = _result_to_costs(result)
            self.portfolio = self.portfolio_mgr.apply_realized_pnl(self.portfolio, realized_pnl)
            self.portfolio = self.portfolio_mgr.apply_costs(self.portfolio, result.total_cost)
            self.portfolio = self.portfolio_mgr.update_turnover(self.portfolio, reduce_lots, result.fill_price, pair_meta)

        elif extended_action == Action.CLOSE_POSITION:
            trade_dir = -self.portfolio.direction
            lots = self.portfolio.total_lots
            result = execute_trade(trade_dir, lots, market_t, next_market, pair_meta, self.config, self.portfolio.direction, self.portfolio.total_lots, self._rng)
            self.portfolio, realized_pnl = self.position_mgr.close_position(self.portfolio, result.fill_price, self.simulator.pip_size, pair_meta)
            cost_breakdown = _result_to_costs(result)
            self.portfolio = self.portfolio_mgr.apply_realized_pnl(self.portfolio, realized_pnl)
            self.portfolio = self.portfolio_mgr.apply_costs(self.portfolio, result.total_cost)
            self.portfolio = self.portfolio_mgr.update_turnover(self.portfolio, lots, result.fill_price, pair_meta)

        elif extended_action == Action.REVERSE_POSITION:
            trade_dir = -self.portfolio.direction
            close_lots = self.portfolio.total_lots
            # Close leg
            result_close = execute_trade(trade_dir, close_lots, market_t, next_market, pair_meta, self.config, self.portfolio.direction, self.portfolio.total_lots, self._rng)
            self.portfolio, realized_pnl = self.position_mgr.close_position(self.portfolio, result_close.fill_price, self.simulator.pip_size, pair_meta)
            self.portfolio = self.portfolio_mgr.apply_realized_pnl(self.portfolio, realized_pnl)
            self.portfolio = self.portfolio_mgr.apply_costs(self.portfolio, result_close.total_cost)
            self.portfolio = self.portfolio_mgr.update_turnover(self.portfolio, close_lots, result_close.fill_price, pair_meta)
            # Open leg
            new_dir = trade_dir
            result_open = execute_trade(new_dir, self.position_mgr.base_lot_size, market_t, next_market, pair_meta, self.config, self.portfolio.direction, self.portfolio.total_lots, self._rng)
            self.portfolio = self.position_mgr.open_position(self.portfolio, new_dir, result_open.fill_price)
            self.portfolio = self.portfolio_mgr.apply_costs(self.portfolio, result_open.total_cost)
            self.portfolio = self.portfolio_mgr.update_turnover(self.portfolio, self.position_mgr.base_lot_size, result_open.fill_price, pair_meta)
            cost_breakdown = _merge_costs(_result_to_costs(result_close), _result_to_costs(result_open))

        # Mark to close_{t+1}
        mark_price = next_market.close
        self.portfolio = self.portfolio_mgr.update_unrealized_pnl(self.portfolio, mark_price, self.simulator.pip_size, pair_meta)
        self.portfolio = self.portfolio_mgr.update_margin(self.portfolio, mark_price, self.simulator.pip_size, pair_meta)
        self.portfolio.last_action = extended_action

        # FIX 1: terminated was never set to True on liquidation — the agent kept stepping on a
        # wiped account, corrupting all subsequent TD targets.
        # FIX 2: Liquidation used execute_trade(..., next_market) which fills at next_market.open.
        # The margin call is triggered by the bar's close price (mark_price) so that is the only
        # price available at the point of liquidation — use it directly without an extra trade call.
        terminated = False
        forced_liquidation_event = False
        if self.risk.check_liquidation(self.portfolio) and self.portfolio.direction != 0:
            liq_price = mark_price
            liq_lots = self.portfolio.total_lots
            self.portfolio, liq_pnl = self.position_mgr.close_position(
                self.portfolio, liq_price, self.simulator.pip_size, pair_meta
            )
            self.portfolio = self.portfolio_mgr.apply_realized_pnl(self.portfolio, liq_pnl)
            liq_tc = compute_transaction_costs(
                lots=liq_lots,
                fill_price=liq_price,
                pip_size=self.simulator.pip_size,
                spread_proxy=next_market.spread_proxy,
                pair_metadata=pair_meta,
                config=tc_cfg,
            )
            self.portfolio = self.portfolio_mgr.apply_costs(self.portfolio, liq_tc["total_cost"])
            cost_breakdown["spread_cost"] += float(liq_tc.get("spread_cost", 0.0))
            cost_breakdown["commission_cost"] += float(liq_tc.get("commission_cost", 0.0))
            cost_breakdown["total_cost"] += float(liq_tc.get("total_cost", 0.0))
            self.portfolio.forced_liquidations += 1
            self.portfolio = self.portfolio_mgr.update_unrealized_pnl(self.portfolio, mark_price, self.simulator.pip_size, pair_meta)
            self.portfolio = self.portfolio_mgr.update_margin(self.portfolio, mark_price, self.simulator.pip_size, pair_meta)
            terminated = True
            forced_liquidation_event = True

        # Advance cursor
        current_market = self.simulator.advance()
        self._step_count += 1

        # Truncation check (keep terminated/truncated mutually exclusive for cleaner episode semantics).
        if terminated:
            truncated = False
        else:
            if self.end_policy == "max_steps":
                if self.max_steps is not None:
                    truncated = bool(self._step_count >= int(self.max_steps))
                else:
                    truncated = not self.simulator.has_next()
            else:
                truncated = not self.simulator.has_next()
                if self.max_steps is not None and self._step_count >= int(self.max_steps):
                    truncated = True

        # Reward
        equity_after = self.portfolio.equity
        reward_breakdown = {}
        if self.reward_engine is not None:
            reward, reward_breakdown = self.reward_engine.compute(
                prev_portfolio,
                self.portfolio,
                extended_action,
                cost_breakdown,
                market_t,
                next_market,
                was_legal=was_legal,
                invalid_action_penalty=actual_penalty,
            )
        else:
            eps = 1e-8
            reward = (equity_after - equity_before) / max(abs(equity_before), eps)

        reward -= actual_penalty

        # Build observation
        final_mask = compute_legal_mask(self.portfolio, current_market, self.risk, self.simplified_mode)
        obs = self._build_obs(market_state=current_market, legal_mask=final_mask)
        info = self._build_info(
            raw_action=raw_action,
            executed_action=extended_action,
            was_legal=was_legal,
            cost_breakdown=cost_breakdown,
            reward_breakdown=reward_breakdown,
            realized_pnl=realized_pnl,
            equity_before=equity_before,
            equity_after=equity_after,
            cash_before=cash_before,
            cash_after=self.portfolio.cash,
            direction_before=direction_before,
            total_lots_before=prev_portfolio.total_lots,
            total_lots_after=self.portfolio.total_lots,
            pyramid_levels_before=prev_portfolio.pyramid_levels,
            martingale_steps_before=prev_portfolio.martingale_steps,
            turnover_before=prev_portfolio.turnover,
            turnover_after=self.portfolio.turnover,
            decision_ts=decision_ts,
            execution_ts=execution_ts,
            forced_liquidation=forced_liquidation_event,
            market_state=current_market,
            legal_mask=final_mask,
        )

        return obs, float(reward), terminated, truncated, info

    def _build_obs(
        self,
        market_state: Optional[Any] = None,
        legal_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        idx = self.simulator.cursor
        start = max(0, idx - self.window_length + 1)
        window = self._feature_matrix[start:idx + 1]
        # Pad if not enough history
        if window.shape[0] < self.window_length:
            pad = np.zeros((self.window_length - window.shape[0], window.shape[1]), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)

        if legal_mask is None:
            market = market_state if market_state is not None else self.simulator.current_state()
            mask = compute_legal_mask(self.portfolio, market, self.risk, self.simplified_mode)
        else:
            mask = legal_mask
        return build_observation(window, self.portfolio, mask, self.config)

    def _build_info(
        self,
        raw_action,
        executed_action,
        was_legal,
        cost_breakdown,
        reward_breakdown,
        realized_pnl=0.0,
        equity_before=None,
        equity_after=None,
        cash_before=None,
        cash_after=None,
        direction_before=None,
        total_lots_before=None,
        total_lots_after=None,
        pyramid_levels_before=None,
        martingale_steps_before=None,
        turnover_before=None,
        turnover_after=None,
        decision_ts=None,
        execution_ts=None,
        forced_liquidation=False,
        market_state=None,
        legal_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        market = market_state if market_state is not None else self.simulator.current_state()
        if legal_mask is None:
            legal_mask = compute_legal_mask(self.portfolio, market, self.risk, self.simplified_mode)
        # FIX: Replace `x or fallback` with `x if x is not None else fallback`. The truthiness
        # pattern silently substitutes a valid 0.0 (fully-wiped equity) with the wrong value.
        return {
            "step": self._step_count,
            "timestamp_decision": decision_ts if decision_ts is not None else market.timestamp,
            "timestamp_execution": execution_ts if execution_ts is not None else market.timestamp,
            "raw_action": raw_action,
            "executed_action": int(executed_action),
            "action_name": ACTION_NAMES.get(int(executed_action), "UNKNOWN"),
            "executed_action_name": ACTION_NAMES.get(int(executed_action), "UNKNOWN"),
            "action_mode": "simplified" if self.simplified_mode else "extended",
            "was_legal": was_legal,
            "invalid_action_policy": self.invalid_action_policy if not was_legal else None,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": self.portfolio.unrealized_pnl,
            "equity_before": equity_before if equity_before is not None else self.portfolio.equity,
            "equity_after": equity_after if equity_after is not None else self.portfolio.equity,
            "cash_before": cash_before if cash_before is not None else self.portfolio.cash,
            "cash_after": cash_after if cash_after is not None else self.portfolio.cash,
            "used_margin": self.portfolio.used_margin,
            "free_margin": self.portfolio.free_margin,
            "cost_breakdown": cost_breakdown,
            "drawdown": self.portfolio.current_drawdown,
            "mask": legal_mask,
            "reward_breakdown": reward_breakdown,
            "forced_liquidation": bool(forced_liquidation),
            "forced_liquidations_count": int(self.portfolio.forced_liquidations),
            "session_label": market.session_label,
            "direction_before": int(direction_before) if direction_before is not None else int(self.portfolio.direction),
            "direction": self.portfolio.direction,
            "total_lots": self.portfolio.total_lots,
            "total_lots_before": float(total_lots_before) if total_lots_before is not None else float(self.portfolio.total_lots),
            "total_lots_after": float(total_lots_after) if total_lots_after is not None else float(self.portfolio.total_lots),
            "pyramid_levels": int(self.portfolio.pyramid_levels),
            "pyramid_levels_before": int(pyramid_levels_before) if pyramid_levels_before is not None else int(self.portfolio.pyramid_levels),
            "martingale_steps": int(self.portfolio.martingale_steps),
            "martingale_steps_before": int(martingale_steps_before) if martingale_steps_before is not None else int(self.portfolio.martingale_steps),
            "turnover_before": float(turnover_before) if turnover_before is not None else float(self.portfolio.turnover),
            "turnover_after": float(turnover_after) if turnover_after is not None else float(self.portfolio.turnover),
            "turnover_delta": (
                float(turnover_after) - float(turnover_before)
                if turnover_before is not None and turnover_after is not None
                else 0.0
            ),
            "equity": self.portfolio.equity,
        }


def _result_to_costs(result) -> Dict[str, float]:
    return {
        "spread_cost": result.spread_cost,
        "commission_cost": result.commission_cost,
        "slippage_cost": result.slippage_cost,
        "rollover_cost": result.rollover_cost,
        "total_cost": result.total_cost,
    }


def _merge_costs(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {k: a.get(k, 0.0) + b.get(k, 0.0) for k in set(a) | set(b)}
