"""Native Market Execution Handler implementing Rosa Roja's ExecutionPort."""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from core.orchestration.rosa_roja.ports.execution_port import ExecutionPort
from core.orchestration.rosa_roja.domain.execution import ExecutionPlan
from core.orchestration.rosa_roja.domain.trajectory import Trajectory

logger = logging.getLogger(__name__)


class BrokerClientProtocol:
    """Minimal broker interface for order dispatch."""
    
    def submit_order(self, symbol: str, side: str, order_type: str, 
                     qty: float, price: Optional[float] = None, 
                     stop_price: Optional[float] = None, 
                     time_in_force: str = "GTC",
                     client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Submit order to broker. Returns order response."""
        ...
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        ...
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders for symbol. Returns count cancelled."""
        ...
    
    def get_position(self, symbol: str) -> float:
        """Get current position size (positive=long, negative=short)."""
        ...
    
    def close_position(self, symbol: str) -> bool:
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
        broker_client: BrokerClientProtocol,
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
        self._active_orders: Dict[str, Dict[str, Any]] = {}

    def dispatch_execution(self, plan: ExecutionPlan) -> bool:
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

        if plan.regime_alert or plan.action == "EMERGENCY_FLUSH":
            self.trigger_emergency_flush(
                reason=plan.veto_details.get("reason", "RegimeAlert_Triggered")
            )
            return False

        if plan.action == "EXECUTE" and plan.chosen_trajectory:
            return self._execute_trajectory_orders(plan)

        logger.warning("Unknown ExecutionPlan action", extra={"action": plan.action})
        return False

    def _execute_trajectory_orders(self, plan: ExecutionPlan) -> bool:
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
            if envelope is None:
                logger.warning("EXECUTE plan missing envelope")
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
            opposite_side = "sell" if side == "buy" else "buy"
            
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
            
            # 1. Entry order (Market)
            entry_order = self._broker.submit_order(
                symbol=self._symbol,
                side=side,
                order_type="market",
                qty=qty,
                time_in_force="IOC",
                client_order_id=f"RR_{terminal_state.step_index}_entry",
            )
            self._track_order(entry_order)
            
            # 2. Stop-loss order
            if stop_pct:
                stop_price = current_price * (1 - stop_pct) if side == "buy" else current_price * (1 + stop_pct)
                sl_order = self._broker.submit_order(
                    symbol=self._symbol,
                    side=opposite_side,
                    order_type="stop",
                    qty=qty,
                    stop_price=round(stop_price, 2),
                    time_in_force="GTC",
                    client_order_id=f"RR_{terminal_state.step_index}_sl",
                )
                self._track_order(sl_order)
            
            # 3. Take-profit order
            if target_pct:
                tp_price = current_price * (1 + target_pct) if side == "buy" else current_price * (1 - target_pct)
                tp_order = self._broker.submit_order(
                    symbol=self._symbol,
                    side=opposite_side,
                    order_type="limit",
                    qty=qty,
                    price=round(tp_price, 2),
                    time_in_force="GTC",
                    client_order_id=f"RR_{terminal_state.step_index}_tp",
                )
                self._track_order(tp_order)
            
            # 4. Invalidation timer scheduling (for external handler)
            if invalidation_step:
                self._schedule_invalidation_check(invalidation_step, terminal_state.step_index)
            
            return True
            
        except Exception as e:
            logger.error("Failed to dispatch execution plan", extra={"error": str(e)})
            return False

    def _get_reference_price(self) -> float:
        """Get current market reference price (midpoint or last trade)."""
        # In production, this would fetch from market data feed
        # For now, return a placeholder
        return 100.0

    def _track_order(self, order_response: Dict[str, Any]) -> None:
        """Track active order for potential cancellation."""
        order_id = order_response.get("order_id") or order_response.get("client_order_id")
        if order_id:
            self._active_orders[order_id] = order_response

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

    def trigger_emergency_flush(self, reason: str) -> None:
        """
        Triggers emergency cancellation and risk protocol.
        
        Called when:
        - Module 1 detects regime change (Mahalanobis outlier)
        - Module 3 hard-gating vetoes all trajectories
        - External risk limits breached
        
        Args:
            reason: Human-readable reason for emergency action.
        """
        logger.warning("EMERGENCY FLUSH TRIGGERED", extra={"reason": reason, "symbol": self._symbol})
        
        # Cancel all active tracked orders
        for order_id in list(self._active_orders.keys()):
            try:
                self._broker.cancel_order(order_id)
                del self._active_orders[order_id]
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}", extra={"error": str(e)})
        
        # Cancel any remaining orders on broker for symbol
        cancelled = self._broker.cancel_all_orders(symbol=self._symbol)
        logger.info(f"Emergency flush: cancelled {cancelled} orders for {self._symbol}")
        
        # Flatten position
        position = self._broker.get_position(self._symbol)
        if position != 0:
            self._broker.close_position(self._symbol)
            logger.info(f"Emergency flatten: closed position of {position} for {self._symbol}")

    def update_equity(self, new_equity: float) -> None:
        """Update account equity for position sizing."""
        self._equity = new_equity

    def get_active_orders(self) -> Dict[str, Dict[str, Any]]:
        """Return copy of active orders."""
        return self._active_orders.copy()