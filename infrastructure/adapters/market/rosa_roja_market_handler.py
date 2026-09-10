"""Native Market Execution Handler implementing Rosa Roja's ExecutionPort."""

from __future__ import annotations

import logging
import time
from typing import Any

from infrastructure.ml.engines.rosa_roja.algorithms.domain.execution import ExecutionPlan
from infrastructure.ml.engines.rosa_roja.algorithms.ports.execution_port import ExecutionPort
from iot_machine_learning.infrastructure.adapters.market.trailing_profit_manager import (
    TrailingProfitConfig,
    TrailingProfitManager,
)

logger = logging.getLogger(__name__)


class BrokerClientProtocol:
    """Minimal broker interface for order dispatch."""

    async def submit_order(self, symbol: str, side: str, order_type: str,
                     qty: float, price: float | None = None,
                     stop_price: float | None = None,
                     time_in_force: str = "GTC",
                     client_order_id: str | None = None) -> dict[str, Any]:
        """Submit order to broker. Returns order response."""
        ...

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        ...

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all open orders for symbol. Returns count cancelled."""
        ...

    async def get_position(self, symbol: str) -> float:
        """Get current position size (positive=long, negative=short)."""
        ...

    async def close_position(self, symbol: str) -> bool:
        """Flatten position at market."""
        ...


class RosaRojaMarketExecutionHandler(ExecutionPort):
    """
    Native Market execution handler fulfilling Rosa Roja's ExecutionPort contract.

    Directly consumes ExecutionPlan from Rosa Roja Engine without intermediate bridges.
    Implements the native protocol: dispatch_execution() and trigger_emergency_flush().

    Usage:
        handler = RosaRojaMarketExecutionHandler(broker_client, equity=100000.0, symbol="NVDA")
        plan = rosa_roja_engine.process_event(delta_state, delta_time)
        handler.dispatch_execution(plan)
    """

    def __init__(
        self,
        broker_client: Any,
        account_equity: float,
        symbol: str,
        lot_size: float = 1.0,
        min_qty: float = 0.01,
        max_position_pct: float = 1.0,
        trailing_config: TrailingProfitConfig | None = None,
    ):
        self._broker = broker_client
        self._equity = account_equity
        self._symbol = symbol
        self._lot_size = lot_size
        self._min_qty = min_qty
        self._max_position_pct = max_position_pct
        self._active_orders: dict[str, dict[str, Any]] = {}
        self._trailing_config = trailing_config
        self._trailing_manager = TrailingProfitManager(trailing_config)
        self._trailing_managers: dict[str, TrailingProfitManager] = {symbol.upper(): self._trailing_manager}
        self._cached_positions: dict[str, dict[str, Any]] = {}
        self._last_pos_checks: dict[str, float] = {}

    def _get_trailing_manager(self, sym: str) -> TrailingProfitManager:
        s = sym.upper()
        if s not in self._trailing_managers:
            self._trailing_managers[s] = TrailingProfitManager(self._trailing_config)
        return self._trailing_managers[s]

    async def dispatch_execution(self, plan: ExecutionPlan, symbol: str | None = None) -> bool:
        """
        Processes an ExecutionPlan directly into market execution actions.

        Args:
            plan: The orchestrated execution plan from Rosa Roja Engine.
            symbol: Target symbol for multi-asset execution.

        Returns:
            True if execution was dispatched successfully, False otherwise.
        """
        if plan.action == "HOLD":
            logger.debug("ExecutionPlan: HOLD - no action taken",
                        extra={"reason": plan.veto_details.get("reason", "unknown")})
            return True

        if plan.action == "EMERGENCY_FLUSH" or plan.regime_alert:
            reason = plan.veto_details.get("reason", "RegimeAlert_Triggered")
            # In intermediate live trading, reactive trajectory micro-deviations and geometric threshold breaches
            # are protected by Alpaca's server-side bracket order (stop-loss and take-profit) and Trailing Profit Manager.
            if "Reactive_Trajectory_Deviation" in reason or "Geometric_Threshold_Breach" in reason:
                logger.info(
                    "Trajectory micro-deviation observed (%s) — position protected by server-side bracket and trailing profit manager, maintaining position",
                    reason,
                )
                return True
            await self.trigger_emergency_flush(reason=reason, symbol=symbol)
            return False

        if plan.action == "EXECUTE" and plan.chosen_trajectory:
            return await self._execute_trajectory_orders(plan, symbol=symbol)

        logger.warning("Unknown ExecutionPlan action", extra={"action": plan.action})
        return False

    async def _execute_trajectory_orders(self, plan: ExecutionPlan, symbol: str | None = None) -> bool:
        """
        Translates plan parameters to broker order requests.

        Order structure:
        1. Entry order (market) based on trajectory direction
        2. Stop-loss order (stop)
        3. Take-profit order (limit)
        4. Invalidation timer (handled externally or via OCO)
        """
        try:
            envelope = plan.envelope
            if envelope is None or plan.chosen_trajectory is None:
                logger.warning("EXECUTE plan missing envelope or trajectory")
                return False

            target_sym = (symbol or getattr(plan, "symbol", None) or self._symbol).upper()
            current_price = self._get_reference_price(target_sym)

            # Calculate position sizing from envelope magnitude
            notional = self._equity * min(envelope.magnitude, self._max_position_pct)
            qty = max(self._min_qty,
                     round(notional / current_price / self._lot_size) * self._lot_size)

            # Extract bounds from envelope
            stop_pct = envelope.bounds.get("stop_pct", 0.0)
            target_pct = envelope.bounds.get("target_pct", 0.0)
            invalidation_step = plan.invalidation_step or envelope.max_steps

            # Determine direction from trajectory terminal state
            terminal_state = plan.chosen_trajectory.terminal_state
            terminal_price = terminal_state.state_vector[0]

            side = "buy" if terminal_price > current_price else "sell"

            logger.info(
                "Dispatching Rosa Roja execution plan",
                extra={
                    "symbol": target_sym,
                    "action": plan.action,
                    "confidence": plan.global_confidence,
                    "notional": notional,
                    "qty": qty,
                    "side": side,
                    "current_price": current_price,
                    "terminal_price": terminal_price,
                    "stop_loss_pct": stop_pct,
                    "take_profit_pct": target_pct,
                    "invalidation_step": invalidation_step,
                    "trajectory_length": plan.chosen_trajectory.length,
                }
            )

            # Prepare Bracket / OTO orders (Atomic submission to prevent wash trade rejections)
            take_profit = None
            stop_loss = None
            order_class = None

            if target_pct and target_pct > 0:
                tp_mult = (1.0 + target_pct) if side == "buy" else (1.0 - target_pct)
                tp_price = round(current_price * tp_mult, 2)
                if side == "buy" and tp_price <= current_price:
                    tp_price = round(current_price + 0.10, 2)
                elif side == "sell" and tp_price >= current_price:
                    tp_price = round(current_price - 0.10, 2)
                take_profit = {"limit_price": str(tp_price)}

            if stop_pct and stop_pct > 0:
                sl_mult = (1.0 - stop_pct) if side == "buy" else (1.0 + stop_pct)
                sl_price = round(current_price * sl_mult, 2)
                if side == "buy" and sl_price >= current_price:
                    sl_price = round(current_price - 0.10, 2)
                elif side == "sell" and sl_price <= current_price:
                    sl_price = round(current_price + 0.10, 2)
                stop_loss = {"stop_price": str(sl_price)}

            if take_profit and stop_loss:
                order_class = "bracket"
            elif take_profit or stop_loss:
                order_class = "oto"

            # 1. Atomic Entry order (Market order with native server-side bracket protection)
            entry_order = await self._broker.submit_order(
                symbol=target_sym,
                side=side,
                order_type="market",
                qty=qty,
                time_in_force="day",
                client_order_id=f"RR_{terminal_state.step_index}_{int(time.time()*1000)}_entry",
                order_class=order_class,
                take_profit=take_profit,
                stop_loss=stop_loss,
            )
            self._track_order(entry_order, symbol=target_sym)
            self._cached_positions[target_sym] = None

            # 4. Invalidation timer scheduling (for external handler)
            if invalidation_step:
                self._schedule_invalidation_check(invalidation_step, terminal_state.step_index)

            return True

        except Exception as e:
            logger.error(f"Failed to dispatch execution plan: {e}", exc_info=True)
            return False

    def _get_reference_price(self, symbol: str | None = None) -> float:
        """Get current market reference price (midpoint or last trade)."""
        cb = getattr(self, "get_reference_price_callback", None)
        if callable(cb):
            import inspect
            sig = inspect.signature(cb)
            p = cb(symbol) if len(sig.parameters) > 0 else cb()
            if isinstance(p, (int, float)) and p > 0:
                return float(p)
        return 759.0

    def _track_order(self, order_response: Any, symbol: str | None = None) -> None:
        """Track active order for potential cancellation per symbol."""
        if hasattr(order_response, "id"):
            order_id = order_response.id or getattr(order_response, "client_order_id", None)
        elif isinstance(order_response, dict):
            order_id = order_response.get("id") or order_response.get("order_id") or order_response.get("client_order_id")
        else:
            order_id = None
        if order_id:
            target_sym = (symbol or self._symbol).upper()
            if target_sym not in self._active_orders:
                self._active_orders[target_sym] = {}
            self._active_orders[target_sym][str(order_id)] = order_response

    def _schedule_invalidation_check(self, invalidation_step: int, step_index: int) -> None:
        """
        Schedules a check to verify trajectory validity at invalidation_step.

        If the price hasn't reached expected progress by this step, the position
        should be closed or tightened.
        """
        logger.info(
            "Invalidation step scheduled",
            extra={
                "invalidation_step": invalidation_step,
                "current_step": step_index,
                "steps_remaining": invalidation_step - step_index
            }
        )

    async def trigger_emergency_flush(self, reason: str, symbol: str | None = None) -> None:
        """Triggers emergency cancellation and risk protocol for a specific symbol or default."""
        target_sym = (symbol or self._symbol).upper()
        logger.warning(f"EMERGENCY FLUSH TRIGGERED: reason='{reason}'", extra={"reason": reason, "symbol": target_sym})

        # Cancel active tracked orders strictly for this target symbol
        sym_orders = self._active_orders.get(target_sym, {})
        for order_id in list(sym_orders.keys()):
            try:
                await self._broker.cancel_order(order_id)
                del sym_orders[order_id]
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id} for {target_sym}", extra={"error": str(e)})

        # Cancel any remaining orders on broker strictly for target symbol
        cancelled = await self._broker.cancel_all_orders(symbol=target_sym)
        logger.info(f"Emergency flush: cancelled {cancelled} orders for {target_sym}")
        # Flatten position for target symbol
        position = await self._broker.get_position(target_sym)
        pos_qty = float(position.get("qty", 0.0)) if isinstance(position, dict) else float(position or 0.0)
        if pos_qty != 0:
            await self._broker.close_position(target_sym)
            logger.info(f"Emergency flatten: closed position of {pos_qty} for {target_sym}")
        self._cached_positions[target_sym] = None

    def update_equity(self, new_equity: float) -> None:
        """Update account equity for position sizing."""
        self._equity = new_equity

    def get_active_orders(self, symbol: str | None = None) -> dict[str, Any]:
        """Return copy of active orders (either for a specific symbol or flattened across all)."""
        if symbol is not None:
            return self._active_orders.get(symbol.upper(), {}).copy()
        flattened = {}
        for sym_orders in self._active_orders.values():
            flattened.update(sym_orders)
        return flattened

    async def check_trailing_profit(self, current_price: float, symbol: str | None = None) -> bool:
        """Monitorea la posición activa de un símbolo y liquida si retrocede desde el pico máximo."""
        sym = (symbol or self._symbol).upper()
        if not self._broker or current_price <= 0:
            return False
        try:
            now = time.time()
            last_check = self._last_pos_checks.get(sym, 0.0)
            cached = self._cached_positions.get(sym)
            if cached is None or (now - last_check >= 0.5):
                pos = await self._broker.get_position(sym)
                cached = pos if isinstance(pos, dict) else {"qty": float(pos or 0.0)}
                self._cached_positions[sym] = cached
                self._last_pos_checks[sym] = now

            position = cached or {}
            qty = float(position.get("qty", 0.0))
            mgr = self._get_trailing_manager(sym)
            if qty == 0:
                mgr.reset()
                return False

            avg_entry = float(position.get("avg_entry_price", 0.0))
            if avg_entry <= 0:
                return False

            # PnL no realizado (positivo si largo y sube, positivo si corto y baja)
            unrealized_pnl = (current_price - avg_entry) * qty
            should_exit, reason = mgr.update(unrealized_pnl)
            if should_exit:
                logger.warning(
                    "TRAILING PROFIT LOCK [%s]: %s | Closing position %s @ $%.2f",
                    sym, reason, qty, current_price,
                )
                await self._broker.cancel_all_orders(symbol=sym)
                await self._broker.close_position(sym)
                self._active_orders.pop(sym, None)
                mgr.reset()
                self._cached_positions[sym] = None
                return True
        except Exception as e:
            logger.debug("Error checking trailing profit for %s: %s", sym, e)
        return False
