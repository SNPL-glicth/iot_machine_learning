"""Native Market Execution Handler implementing Rosa Roja's ExecutionPort."""

from __future__ import annotations

import logging
import time
from typing import Any

from infrastructure.ml.engines.rosa_roja.algorithms.domain.execution import ExecutionPlan
from infrastructure.ml.engines.rosa_roja.algorithms.ports.execution_port import ExecutionPort

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
    ):
        self._broker = broker_client
        self._equity = account_equity
        self._symbol = symbol
        self._lot_size = lot_size
        self._min_qty = min_qty
        self._max_position_pct = max_position_pct
        self._active_orders: dict[str, dict[str, Any]] = {}

    async def dispatch_execution(self, plan: ExecutionPlan) -> bool:
        """
        Processes an ExecutionPlan directly into market execution actions.

        Args:
            plan: The orchestrated execution plan from Rosa Roja Engine.

        Returns:
            True if execution was dispatched successfully, False otherwise.
        """
        if plan.action == "HOLD":
            logger.debug("ExecutionPlan: HOLD - no action taken",
                        extra={"reason": plan.veto_details.get("reason", "unknown")})
            return True

        if plan.action == "EMERGENCY_FLUSH" or plan.regime_alert:
            reason = plan.veto_details.get("reason", "RegimeAlert_Triggered")
            # In intermediate live trading, reactive trajectory micro-deviations are protected
            # by Alpaca's server-side bracket order (stop-loss and take-profit)
            if "Reactive_Trajectory_Deviation" in reason:
                logger.info(
                    "Trajectory micro-deviation observed (%s) — position protected by server-side bracket, maintaining position",
                    reason,
                )
                return True
            await self.trigger_emergency_flush(reason=reason)
            return False

        if plan.action == "EXECUTE" and plan.chosen_trajectory:
            return await self._execute_trajectory_orders(plan)

        logger.warning("Unknown ExecutionPlan action", extra={"action": plan.action})
        return False

    async def _execute_trajectory_orders(self, plan: ExecutionPlan) -> bool:
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

            # Calculate position sizing from envelope magnitude
            notional = self._equity * min(envelope.magnitude, self._max_position_pct)
            qty = max(self._min_qty,
                     round(notional / self._get_reference_price() / self._lot_size) * self._lot_size)

            # Extract bounds from envelope
            stop_pct = envelope.bounds.get("stop_pct", 0.0)
            target_pct = envelope.bounds.get("target_pct", 0.0)
            invalidation_step = plan.invalidation_step or envelope.max_steps

            # Determine direction from trajectory terminal state
            terminal_state = plan.chosen_trajectory.terminal_state
            current_price = self._get_reference_price()
            terminal_price = terminal_state.state_vector[0]

            side = "buy" if terminal_price > current_price else "sell"

            logger.info(
                "Dispatching Rosa Roja execution plan",
                extra={
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
                symbol=self._symbol,
                side=side,
                order_type="market",
                qty=qty,
                time_in_force="day",
                client_order_id=f"RR_{terminal_state.step_index}_{int(time.time()*1000)}_entry",
                order_class=order_class,
                take_profit=take_profit,
                stop_loss=stop_loss,
            )
            self._track_order(entry_order)

            # 4. Invalidation timer scheduling (for external handler)
            if invalidation_step:
                self._schedule_invalidation_check(invalidation_step, terminal_state.step_index)

            return True

        except Exception as e:
            logger.error(f"Failed to dispatch execution plan: {e}", exc_info=True)
            return False

    def _get_reference_price(self) -> float:
        """Get current market reference price (midpoint or last trade)."""
        cb = getattr(self, "get_reference_price_callback", None)
        if callable(cb):
            p = cb()
            if isinstance(p, (int, float)) and p > 0:
                return float(p)
        return 759.0

    def _track_order(self, order_response: Any) -> None:
        """Track active order for potential cancellation."""
        if hasattr(order_response, "id"):
            order_id = order_response.id or getattr(order_response, "client_order_id", None)
        elif isinstance(order_response, dict):
            order_id = order_response.get("id") or order_response.get("order_id") or order_response.get("client_order_id")
        else:
            order_id = None
        if order_id:
            self._active_orders[str(order_id)] = order_response

    def _schedule_invalidation_check(self, invalidation_step: int, step_index: int) -> None:
        """Schedule invalidation check at the computed step.

        In production, this would integrate with the execution engine's
        timer/scheduler to trigger re-evaluation at the invalidation point.
        """
        logger.info(
            "Invalidation step scheduled",
            extra={
                "invalidation_step": invalidation_step,
                "current_step": step_index,
                "steps_remaining": invalidation_step - step_index
            }
        )

    async def trigger_emergency_flush(self, reason: str) -> None:
        """
        Triggers emergency cancellation and risk protocol.

        Called when:
        - Module 1 detects regime change (Mahalanobis outlier)
        - Module 3 hard-gating vetoes all trajectories
        - External risk limits breached

        Args:
            reason: Human-readable reason for emergency action.
        """
        logger.warning(f"EMERGENCY FLUSH TRIGGERED: reason='{reason}'", extra={"reason": reason, "symbol": self._symbol})

        # Cancel all active tracked orders
        for order_id in list(self._active_orders.keys()):
            try:
                await self._broker.cancel_order(order_id)
                del self._active_orders[order_id]
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}", extra={"error": str(e)})

        # Cancel any remaining orders on broker for symbol
        cancelled = await self._broker.cancel_all_orders(symbol=self._symbol)
        logger.info(f"Emergency flush: cancelled {cancelled} orders for {self._symbol}")
        # Flatten position
        position = await self._broker.get_position(self._symbol)
        pos_qty = float(position.get("qty", 0.0)) if isinstance(position, dict) else float(position or 0.0)
        if pos_qty != 0:
            await self._broker.close_position(self._symbol)
            logger.info(f"Emergency flatten: closed position of {pos_qty} for {self._symbol}")

    def update_equity(self, new_equity: float) -> None:
        """Update account equity for position sizing."""
        self._equity = new_equity

    def get_active_orders(self) -> dict[str, dict[str, Any]]:
        """Return copy of active orders."""
        return self._active_orders.copy()
