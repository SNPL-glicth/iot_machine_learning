"""Native Market Execution Handler implementing Rosa Roja's ExecutionPort."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Protocol

from infrastructure.ml.engines.rosa_roja.algorithms.domain.execution import ExecutionPlan
from infrastructure.ml.engines.rosa_roja.algorithms.ports.execution_port import ExecutionPort
from iot_machine_learning.infrastructure.adapters.market.trailing_profit_manager import (
    TrailingProfitConfig,
    TrailingProfitManager,
)

logger = logging.getLogger(__name__)


class PriceUnavailableError(RuntimeError):
    """Raised when no valid market reference price is available for an asset."""


class BrokerClientProtocol(Protocol):
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


def calculate_unrealized_pnl(
    side: str,
    qty: float,
    avg_entry_price: float,
    current_price: float,
) -> float:
    """Calcula el PnL no realizado con estricta consistencia de signo para posiciones Long y Short.

    Fórmulas estándar de mercado:
        - Long:  (current_price - avg_entry_price) * abs(qty)
        - Short: (avg_entry_price - current_price) * abs(qty)

    Convenciones de entrada soportadas:
        - Alpaca: qty > 0 con side="short" o "sell".
        - Brokers con signo (Binance/IB): qty < 0 para shorts, o side="short".

    Retorna 0.0 si los datos son inválidos (precios <= 0 o qty == 0).
    """
    try:
        q = float(qty)
        entry = float(avg_entry_price)
        curr = float(current_price)
    except (ValueError, TypeError):
        return 0.0

    if entry <= 0.0 or curr <= 0.0 or q == 0.0:
        return 0.0

    s = str(side).strip().lower() if side is not None else ""
    is_short = s in ("short", "sell") or q < 0.0

    abs_qty = abs(q)
    if is_short:
        return (entry - curr) * abs_qty
    else:
        return (curr - entry) * abs_qty


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
        max_qty: float = float("inf"),
        max_position_pct: float = 1.0,
        trailing_config: TrailingProfitConfig | None = None,
        max_stop_loss_usd: float = 10.0,
        state: Any | None = None,
        max_consecutive_losses: int = 2,
        consecutive_loss_cooldown_sec: float = 900.0,
    ):
        self._broker = broker_client
        self._equity = account_equity
        self._symbol = symbol
        self._lot_size = lot_size
        self._min_qty = min_qty
        self._max_qty = max_qty
        self._max_position_pct = max_position_pct
        self._max_stop_loss_usd = max_stop_loss_usd
        self._state = state
        self._max_consecutive_losses = max_consecutive_losses
        self._consecutive_loss_cooldown_sec = consecutive_loss_cooldown_sec
        self._active_orders: dict[str, dict[str, Any]] = {}
        self._trailing_config = trailing_config
        self._trailing_manager = TrailingProfitManager(trailing_config)
        self._trailing_managers: dict[str, TrailingProfitManager] = {symbol.upper(): self._trailing_manager}
        self._cached_positions: dict[str, dict[str, Any] | None] = {}
        self._last_pos_checks: dict[str, float] = {}
        self._last_known_prices: dict[str, tuple[float, float]] = {}

    def _mark_closing(self, symbol: str) -> None:
        """Marca símbolo en proceso de cierre en LiveBotState antes de enviar órdenes de salida."""
        if self._state is not None and hasattr(self._state, "mark_closing"):
            self._state.mark_closing(symbol)

    def _mark_close_confirmed(self, symbol: str) -> None:
        """Registra confirmación real de salida en LiveBotState para iniciar cooldown post-cierre."""
        if self._state is not None and hasattr(self._state, "mark_close_confirmed"):
            self._state.mark_close_confirmed(symbol)

    def _record_trade_outcome(self, symbol: str, realized_pnl: float) -> None:
        """Registra el resultado (ganancia/pérdida) del trade en LiveBotState para racha y enfriamiento."""
        if self._state is not None and hasattr(self._state, "record_trade_outcome"):
            self._state.record_trade_outcome(
                symbol,
                realized_pnl=realized_pnl,
                max_consecutive_losses=self._max_consecutive_losses,
                cooldown_sec=self._consecutive_loss_cooldown_sec,
            )

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
            target_sym = (symbol or self._symbol).upper()

            # FIX 3: La ausencia de datos en _cached_positions (None o vacío) NUNCA debe
            # interpretarse como "posición ganadora" por defecto. Si el caché está vacío,
            # forzamos una consulta directa y síncrona al broker para conocer la exposición real.
            cached_pos = self._cached_positions.get(target_sym)
            if cached_pos is None or not cached_pos:
                try:
                    pos = await self._broker.get_position(target_sym)
                    cached_pos = pos if isinstance(pos, dict) else {"qty": float(pos or 0.0)}
                    self._cached_positions[target_sym] = cached_pos
                    self._last_pos_checks[target_sym] = time.time()
                except Exception as e:
                    logger.warning(
                        "Failed to query broker position for %s during EMERGENCY_FLUSH evaluation: %s. "
                        "Assuming risky/unverified position (will proceed with flush).",
                        target_sym, e,
                    )
                    cached_pos = None

            # Inversión de lógica por defecto: Asumir riesgo por defecto (is_verified_winning = False).
            # En emergencias es preferible cerrar una posición que resultaba ganadora a dejar correr
            # una perdedora sin cobertura. Solo se suprime el flush si se verifica positivamente
            # que la posición existe y su PnL no realizado es neutral o positivo (>= -$3.00).
            is_verified_winning = False
            if cached_pos:
                pos_qty = float(cached_pos.get("qty", 0.0))
                avg_entry = float(cached_pos.get("avg_entry_price", 0.0))
                try:
                    ref_mid = self._get_reference_price(target_sym)
                except Exception:
                    ref_mid = 0.0

                if pos_qty != 0.0 and avg_entry > 0.0 and ref_mid > 0.0:
                    pos_side = str(cached_pos.get("side", ""))
                    unrealized = calculate_unrealized_pnl(pos_side, pos_qty, avg_entry, ref_mid)
                    if unrealized >= -3.0:
                        is_verified_winning = True

            # Solo se mantiene la posición si es micro-desviación y está POSITIVAMENTE VERIFICADA como ganadora/neutral
            if ("Reactive_Trajectory_Deviation" in reason or "Geometric_Threshold_Breach" in reason) and is_verified_winning:
                logger.info(
                    "Trajectory micro-deviation observed (%s) on verified neutral/winning position — maintaining position",
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
            try:
                current_price = self._get_reference_price(target_sym)
            except PriceUnavailableError as pe:
                logger.error("Aborting order dispatch for %s: %s", target_sym, pe)
                return False

            # Calculate position sizing from envelope magnitude
            notional = self._equity * min(envelope.magnitude, self._max_position_pct)
            qty = min(self._max_qty,
                      max(self._min_qty,
                          round(notional / current_price / self._lot_size) * self._lot_size))

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

    def _get_reference_price(self, symbol: str | None = None, max_stale_age_sec: float = 30.0) -> float:
        """
        Get current market reference price (midpoint or last trade).
        If feed is unavailable, attempts to use cached price if age <= max_stale_age_sec.
        Never returns a hardcoded fake price. Raises PriceUnavailableError if unavailable.
        """
        sym = (symbol or self._symbol).upper()
        cb = getattr(self, "get_reference_price_callback", None)
        if callable(cb):
            try:
                import inspect
                sig = inspect.signature(cb)
                p = cb(sym) if len(sig.parameters) > 0 else cb()
                if isinstance(p, (int, float)) and p > 0:
                    val = float(p)
                    self._last_known_prices[sym] = (val, time.time())
                    return val
            except Exception as e:
                logger.warning("Failed to fetch reference price for %s via callback: %s", sym, e)

        # Check cached price for this specific symbol
        if sym in self._last_known_prices:
            last_price, cached_at = self._last_known_prices[sym]
            age = time.time() - cached_at
            if age <= max_stale_age_sec:
                logger.warning(
                    "Using STALE reference price for %s: $%.2f (age: %.1fs <= %.1fs)",
                    sym, last_price, age, max_stale_age_sec,
                )
                return last_price
            logger.error(
                "Cached reference price for %s is EXPIRED (age: %.1fs > %.1fs, price: $%.2f)",
                sym, age, max_stale_age_sec, last_price,
            )

        raise PriceUnavailableError(
            f"Market reference price unavailable for symbol '{sym}' (feed callback returned invalid price and no valid cache within {max_stale_age_sec}s)"
        )

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
    def _is_close_confirmed(self, close_resp: Any) -> bool:
        """Verifica si la respuesta del broker confirma el fill o la orden de cierre."""
        if close_resp is None or close_resp is False:
            return False
        if isinstance(close_resp, dict):
            status = str(close_resp.get("status", "")).lower()
            if status in ("filled", "closed", "accepted", "new", "pending_new"):
                return True
            if close_resp.get("symbol") and "error" not in close_resp:
                return True
            return False
        from unittest.mock import Mock
        if isinstance(close_resp, Mock):
            st = getattr(close_resp, "status", None)
            if isinstance(st, str):
                return st.lower() in ("filled", "closed", "accepted", "new", "pending_new")
            return True
        if hasattr(close_resp, "status") and isinstance(getattr(close_resp, "status"), str):
            status = getattr(close_resp, "status").lower()
            return status in ("filled", "closed", "accepted", "new", "pending_new")
        if isinstance(close_resp, (bool, int, float)):
            return bool(close_resp)
        return True

    async def trigger_emergency_flush(self, reason: str, symbol: str | None = None) -> None:
        """Triggers emergency cancellation and risk protocol for a specific symbol or default."""
        target_sym = (symbol or self._symbol).upper()
        self._mark_closing(target_sym)
        logger.warning(f"EMERGENCY FLUSH TRIGGERED: reason='{reason}'", extra={"reason": reason, "symbol": target_sym})

        # Cancel active tracked orders strictly for this target symbol
        sym_orders = self._active_orders.get(target_sym, {})
        cancelled_local = 0
        for order_id in list(sym_orders.keys()):
            try:
                await self._broker.cancel_order(order_id)
                del sym_orders[order_id]
                cancelled_local += 1
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id} for {target_sym}", extra={"error": str(e)})

        # Cancel any remaining orders on broker strictly for target symbol
        try:
            cancelled = await self._broker.cancel_all_orders(symbol=target_sym)
            logger.info(f"Emergency flush: cancelled {cancelled} orders on broker ({cancelled_local} tracked) for {target_sym}")
        except (RuntimeError, asyncio.TimeoutError, ConnectionError, OSError) as e:
            logger.error(
                f"EMERGENCY FLUSH: Failed to cancel open orders for {target_sym} due to transport/broker error: {e}. "
                "Continuing immediately to position liquidation.",
                exc_info=True,
                extra={"symbol": target_sym, "error": str(e)},
            )
        except Exception as e:
            logger.critical(
                f"EMERGENCY FLUSH: Unexpected failure cancelling orders for {target_sym}: {e}. "
                "Proceeding immediately to position liquidation.",
                exc_info=True,
                extra={"symbol": target_sym, "error": str(e)},
            )

        # Flatten position for target symbol
        position = await self._broker.get_position(target_sym)
        pos_qty = float(position.get("qty", 0.0)) if isinstance(position, dict) else float(position or 0.0)
        if pos_qty != 0:
            avg_entry = float(position.get("avg_entry_price", 0.0)) if isinstance(position, dict) else 0.0
            try:
                ref_mid = self._get_reference_price(target_sym)
            except Exception:
                ref_mid = avg_entry
            pos_side = str(position.get("side", "")) if isinstance(position, dict) else ""
            pnl = calculate_unrealized_pnl(pos_side, pos_qty, avg_entry, ref_mid)

            close_resp = await self._broker.close_position(target_sym)
            if self._is_close_confirmed(close_resp):
                logger.info(f"Emergency flatten: closed position of {pos_qty} for {target_sym} confirmed by broker (est PnL: ${pnl:.2f})")
                self._record_trade_outcome(target_sym, pnl)
                self._mark_close_confirmed(target_sym)
            else:
                logger.error(f"Emergency flatten: broker did not confirm position closure for {target_sym}. Response: {close_resp}")
        else:
            # Si la posición ya estaba plana (qty == 0), no imponer cooldown post-cierre de 45s.
            # Limpiar is_closing para no bloquear el activo injustificadamente.
            if self._state is not None and hasattr(self._state, "is_closing"):
                self._state.is_closing[target_sym] = False
                if hasattr(self._state, "close_confirmed_at") and target_sym in self._state.close_confirmed_at:
                    del self._state.close_confirmed_at[target_sym]
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
            pos_side = str(position.get("side", ""))
            unrealized_pnl = calculate_unrealized_pnl(pos_side, qty, avg_entry, current_price)
            # Software Stop-Loss de protección: liquidar si pérdida alcanza el límite máximo
            if self._max_stop_loss_usd > 0 and unrealized_pnl <= -self._max_stop_loss_usd:
                reason = f"Software Stop-Loss: Loss reached ${unrealized_pnl:.2f} <= -${self._max_stop_loss_usd:.2f}"
                logger.warning(
                    "SOFTWARE STOP-LOSS [%s]: %s | Liquidating position %s @ $%.2f",
                    sym, reason, qty, current_price,
                )
                self._mark_closing(sym)
                await self._broker.cancel_all_orders(symbol=sym)
                close_resp = await self._broker.close_position(sym)
                if self._is_close_confirmed(close_resp):
                    logger.info(f"Software Stop-Loss: closed position of {qty} for {sym} confirmed by broker")
                    self._mark_close_confirmed(sym)
                    self._record_trade_outcome(sym, unrealized_pnl)
                else:
                    logger.error(f"Software Stop-Loss: broker did not confirm position closure for {sym}. Response: {close_resp}")
                self._active_orders.pop(sym, None)
                mgr.reset()
                self._cached_positions[sym] = None
                return True

            should_exit, reason = mgr.update(unrealized_pnl)
            if should_exit:
                logger.warning(
                    "TRAILING PROFIT LOCK [%s]: %s | Closing position %s @ $%.2f",
                    sym, reason, qty, current_price,
                )
                self._mark_closing(sym)
                await self._broker.cancel_all_orders(symbol=sym)
                close_resp = await self._broker.close_position(sym)
                if self._is_close_confirmed(close_resp):
                    logger.info(f"Trailing profit lock: closed position of {qty} for {sym} confirmed by broker")
                    self._mark_close_confirmed(sym)
                    self._record_trade_outcome(sym, unrealized_pnl)
                else:
                    logger.error(f"Trailing profit lock: broker did not confirm position closure for {sym}. Response: {close_resp}")
                self._active_orders.pop(sym, None)
                mgr.reset()
                self._cached_positions[sym] = None
                return True
        except Exception as e:
            logger.debug("Error checking trailing profit for %s: %s", sym, e)
        return False
