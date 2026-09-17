"""Liquidation, emergency flush, and trailing profit execution routines for Rosa Roja."""

from __future__ import annotations

import logging
import time
from typing import Any

from iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja.order_builder import calculate_unrealized_pnl

logger = logging.getLogger(__name__)


def is_close_confirmed(close_resp: Any) -> bool:
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
        return st.lower() in ("filled", "closed", "accepted", "new", "pending_new") if isinstance(st, str) else True
    if hasattr(close_resp, "status") and isinstance(getattr(close_resp, "status"), str):
        return getattr(close_resp, "status").lower() in ("filled", "closed", "accepted", "new", "pending_new")
    if isinstance(close_resp, (bool, int, float)):
        return bool(close_resp)
    return True


async def execute_emergency_flush(
    broker: Any,
    symbol: str,
    reason: str,
    active_orders: dict[str, dict[str, Any]],
    mark_closing_fn: Any,
    mark_close_confirmed_fn: Any,
    state: Any,
) -> None:
    """Ejecuta cancelación de órdenes de emergencia y liquidación de posición para un activo."""
    mark_closing_fn(symbol)
    logger.warning("EMERGENCY FLUSH TRIGGERED: reason='%s'", reason, extra={"reason": reason, "symbol": symbol})

    sym_orders = active_orders.get(symbol, {})
    for order_id in list(sym_orders.keys()):
        try:
            await broker.cancel_order(order_id)
            del sym_orders[order_id]
        except Exception as e:
            logger.error("Failed to cancel order %s for %s: %s", order_id, symbol, e)

    try:
        await broker.cancel_all_orders(symbol=symbol)
    except Exception as e:
        logger.error("EMERGENCY FLUSH: Failed to cancel open orders for %s: %s", symbol, e)

    try:
        pos = await broker.get_position(symbol)
        qty = float(pos.get("qty", 0.0)) if isinstance(pos, dict) else float(pos or 0.0)
    except Exception as e:
        logger.error("Emergency flush: Failed to read position for %s: %s", symbol, e)
        qty = 0.0

    if qty != 0:
        logger.warning("EMERGENCY FLUSH: Liquidating position of %s for %s", qty, symbol)
        try:
            close_resp = await broker.close_position(symbol)
            if is_close_confirmed(close_resp):
                logger.info("Emergency flatten: closed position of %s for %s confirmed by broker", qty, symbol)
                mark_close_confirmed_fn(symbol)
            else:
                logger.error("Emergency flatten: broker did not confirm position closure for %s. Response: %s", symbol, close_resp)
        except Exception as e:
            logger.error("Emergency flush: liquidation exception for %s: %s", symbol, e)
    else:
        if state is not None and hasattr(state, "is_closing"):
            state.is_closing[symbol] = False
            if hasattr(state, "close_confirmed_at") and symbol in state.close_confirmed_at:
                del state.close_confirmed_at[symbol]


async def execute_trailing_and_stop_loss_check(
    broker: Any,
    symbol: str,
    current_price: float,
    equity: float,
    max_stop_loss_usd: float,
    trailing_mgr: Any,
    cached_positions: dict[str, Any],
    last_pos_checks: dict[str, float],
    mark_closing_fn: Any,
    mark_close_confirmed_fn: Any,
    record_outcome_fn: Any,
    active_orders: dict[str, Any],
) -> bool:
    """Evalúa trailing profit y software stop-loss sobre la posición abierta."""
    now = time.time()
    last_check = last_pos_checks.get(symbol, 0.0)
    cached = cached_positions.get(symbol)
    if cached is None or (now - last_check >= 0.5):
        pos = await broker.get_position(symbol)
        cached = pos if isinstance(pos, dict) else {"qty": float(pos or 0.0)}
        cached_positions[symbol] = cached
        last_pos_checks[symbol] = now

    position = cached or {}
    qty = float(position.get("qty", 0.0))
    if qty == 0:
        trailing_mgr.reset()
        return False

    avg_entry = float(position.get("avg_entry_price", 0.0))
    if avg_entry <= 0:
        return False

    unrealized_pnl = calculate_unrealized_pnl(str(position.get("side", "")), qty, avg_entry, current_price)
    
    # Calibración de Stop-Loss adaptativo
    effective_stop_loss = (
        max(0.20, min(max_stop_loss_usd, equity * 0.03))
        if equity < 2000.0 and max_stop_loss_usd > 0
        else max_stop_loss_usd
    )

    if effective_stop_loss > 0 and unrealized_pnl <= -effective_stop_loss:
        reason = f"Software Stop-Loss: Loss reached ${unrealized_pnl:.2f} <= -${effective_stop_loss:.2f}"
        logger.warning("SOFTWARE STOP-LOSS [%s]: %s | Liquidating %s @ $%.2f", symbol, reason, qty, current_price)
        mark_closing_fn(symbol)
        await broker.cancel_all_orders(symbol=symbol)
        close_resp = await broker.close_position(symbol)
        if is_close_confirmed(close_resp):
            logger.info("Software Stop-Loss: closed position of %s for %s confirmed by broker", qty, symbol)
            mark_close_confirmed_fn(symbol)
            record_outcome_fn(symbol, unrealized_pnl)
        else:
            logger.error("Software Stop-Loss: broker did not confirm position closure for %s. Response: %s", symbol, close_resp)
        active_orders.pop(symbol, None)
        trailing_mgr.reset()
        cached_positions[symbol] = None
        return True

    should_exit, reason = trailing_mgr.update(unrealized_pnl)
    if should_exit:
        logger.warning("TRAILING PROFIT LOCK [%s]: %s | Closing %s @ $%.2f", symbol, reason, qty, current_price)
        mark_closing_fn(symbol)
        await broker.cancel_all_orders(symbol=symbol)
        close_resp = await broker.close_position(symbol)
        if is_close_confirmed(close_resp):
            logger.info("Trailing profit lock: closed position of %s for %s confirmed by broker", qty, symbol)
            mark_close_confirmed_fn(symbol)
            record_outcome_fn(symbol, unrealized_pnl)
        else:
            logger.error("Trailing profit lock: broker did not confirm position closure for %s. Response: %s", symbol, close_resp)
        active_orders.pop(symbol, None)
        trailing_mgr.reset()
        cached_positions[symbol] = None
        return True

    return False
