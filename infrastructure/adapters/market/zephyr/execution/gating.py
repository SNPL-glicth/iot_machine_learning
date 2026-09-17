"""Execution gating, order dispatch, and graceful shutdown routines for LiveBotRunner."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import time
from typing import Any

from iot_machine_learning.infrastructure.adapters.market.zephyr.config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.zephyr.models import (
    ExecutionContext,
    LiveBotState,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.execution.storage import (
    load_state,
    save_state,
)

logger = logging.getLogger(__name__)


def can_execute(
    plan: Any, config: LiveBotConfig, state: LiveBotState, current_price: float,
    symbol: str | None = None, risk_mgr: Any = None, macro_velocity: float = 0.0,
) -> bool:
    """Verifica precondiciones operacionales del broker antes de despachar la orden de ZENIN."""
    if plan.action != "EXECUTE":
        return True
    phi = getattr(state, "last_phi_moe", getattr(plan, "global_confidence", 0.0))
    if phi < getattr(config, "phi_moe_threshold", 0.5):
        return False
    if getattr(state, "last_lambda_t", 0.0) >= getattr(config, "emergency_lambda_threshold", 0.95):
        return False
    if getattr(state, "portfolio_circuit_breaker_tripped", False) or getattr(state, "market_closed", False):
        return False
    if getattr(state, "market_closing_soon", False) or getattr(state, "account_blocked", False):
        return False
    sym = (symbol or config.symbol).upper()
    post_close_cooldown = getattr(config, "post_close_cooldown_sec", 45.0)
    if hasattr(state, "is_symbol_closing") and state.is_symbol_closing(sym, cooldown_sec=post_close_cooldown):
        return False
    if hasattr(state, "is_in_loss_streak_cooldown") and state.is_in_loss_streak_cooldown(sym):
        return False
    side = getattr(plan, "side", "") or ""
    if not side and getattr(plan, "chosen_trajectory", None) is not None:
        traj = plan.chosen_trajectory
        if hasattr(traj, "side"):
            side = str(getattr(traj, "side", ""))
        elif getattr(traj, "terminal_state", None) is not None and hasattr(traj.terminal_state, "state_vector"):
            vec = getattr(traj.terminal_state, "state_vector", None)
            if vec is not None and len(vec) > 0:
                side = "buy" if float(vec[0]) >= 0 else "sell"

    if risk_mgr:
        if hasattr(risk_mgr, "check_macro_velocity") and not risk_mgr.check_macro_velocity(sym, side, macro_velocity)[0]:
            return False
        if hasattr(risk_mgr, "check_correlation_guardrail") and not risk_mgr.check_correlation_guardrail(sym, side, state.positions)[0]:
            return False

    micro_mode_threshold = getattr(config, "micro_mode_threshold_usd", 2000.0)
    equity = getattr(state, "equity", 10000.0)
    if equity < micro_mode_threshold and side.lower() == "sell" and state.get_position(sym) <= 0:
        logger.info("Micro-mode (< $%.0f USD): Short entry vetoed for %s (equity=$%.2f)", micro_mode_threshold, sym, equity)
        return False
    if symbol:
        if state.get_position(symbol) != 0: return False
        if state.active_positions_count >= getattr(config, "max_concurrent_positions", 2): return False
    elif state.active_positions_count > 0:
        return False
    now = time.time()
    cooldown_ms = (config.get_effective_cooldown(state.last_lambda_t) if config.dynamic_cooldown else float(config.cooldown_ms))
    if state.last_execution_time > 0 and (now - state.last_execution_time) * 1000 < cooldown_ms:
        return False
    if state.last_execution_price > 0 and current_price > 0:
        if abs(current_price - state.last_execution_price) / state.last_execution_price < config.min_price_change_pct:
            return False
    return True


def get_current_mid(feed: Any, symbol: str | None = None) -> float:
    """Obtiene mid-price actual del order book o feed para un símbolo."""
    if feed and symbol and hasattr(feed, "get_mid_price") and callable(getattr(feed, "get_mid_price", None)):
        try:
            p = feed.get_mid_price(symbol)
            if p is not None and isinstance(p, (int, float)): return float(p)
        except Exception: pass
    if feed and hasattr(feed, "order_book") and feed.order_book and getattr(feed.order_book, "is_initialized", False):
        try:
            mid = feed.order_book.mid_price
            if mid is not None and isinstance(mid, (int, float)): return float(mid)
        except Exception: pass
    if feed and hasattr(feed, "mid_price") and feed.mid_price is not None:
        try:
            if isinstance(feed.mid_price, (int, float)): return float(feed.mid_price)
        except Exception: pass
    return 0.0


async def log_execution(
    plan: Any, state: LiveBotState, config: LiveBotConfig, audit_path: Path | None,
    execution_history: Any,
) -> None:
    """Registra la ejecución en el historial y en audit log estructurado."""
    if not config.enable_audit_log or not audit_path:
        return
    ctx = ExecutionContext(
        timestamp=time.time(), phi_moe=state.last_phi_moe, lambda_t=state.last_lambda_t, phi_ritmo=state.last_phi_ritmo,
        action=plan.action, side="", qty=0.0, price=0.0, order_type="",
        decision_trace=plan.envelope.metadata.get("decision_trace", {}) if plan.envelope else {}, telemetry_hash="",
    )
    execution_history.append(ctx)
    try:
        log_file = audit_path / f"audit_{time.strftime('%Y%m%d')}.ndjson"
        trace = ctx.decision_trace or {}
        record = {
            "timestamp": float(ctx.timestamp), "phi_moe": float(ctx.phi_moe), "lambda_t": float(ctx.lambda_t),
            "phi_ritmo": float(ctx.phi_ritmo), "action": ctx.action,
        }
        if "risk_engine_shadow" in trace and trace["risk_engine_shadow"]: record["risk_engine_shadow"] = trace["risk_engine_shadow"]
        if "temporal_engine_shadow" in trace and trace["temporal_engine_shadow"]: record["temporal_engine_shadow"] = trace["temporal_engine_shadow"]
        import aiofiles
        async with aiofiles.open(log_file, "a") as f:
            await f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("Failed to write audit log: %s", e)


async def perform_shutdown(
    handler: Any, order_client: Any, feed: Any, account: Any, state: LiveBotState,
    config: LiveBotConfig, state_path: Path | str | None, weaviate_store: Any = None,
) -> None:
    """Ejecuta el protocolo de parada limpia cancelando órdenes vivas y drenando persistencia."""
    logger.info("Starting graceful shutdown sequence...")
    if order_client and hasattr(order_client, "cancel_all_orders"):
        try:
            cancelled = await order_client.cancel_all_orders()
            logger.info("Graceful shutdown: cancelled %d open broker orders", cancelled)
        except Exception as e:
            logger.warning("Error cancelling open orders on shutdown: %s", e)
    if order_client and hasattr(order_client, "close_position") and state:
        positions_to_close = {sym: qty for sym, qty in getattr(state, "positions", {}).items() if qty != 0}
        for sym in positions_to_close:
            try:
                await order_client.close_position(sym)
                logger.info("Graceful shutdown: closed open position for %s", sym)
            except Exception as e:
                logger.warning("Error closing position for %s on shutdown: %s", sym, e)
    if weaviate_store and hasattr(weaviate_store, "flush_and_close"):
        try:
            flushed = await weaviate_store.flush_and_close()
            logger.info("Graceful shutdown: flushed %d buffered Weaviate telemetry items", flushed)
        except Exception as e:
            logger.warning("Error draining Weaviate queue on shutdown: %s", e)
    if feed and hasattr(feed, "disconnect"):
        try: await feed.disconnect()
        except Exception as e: logger.warning("Error disconnecting feed: %s", e)
    if order_client and hasattr(order_client, "close"):
        try:
            res = order_client.close()
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:
            logger.warning("Error closing order_client: %s", e)
    if account and hasattr(account, "close"):
        try: await account.close()
        except Exception as e: logger.warning("Error closing account: %s", e)
    await save_state(state, state_path)
    logger.info("Shutdown complete. State saved.")
