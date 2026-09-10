"""Execution logic, risk checks, auditing and persistence for LiveBotRunner."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional

from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.live_runner_models import ExecutionContext, LiveBotState

logger = logging.getLogger(__name__)


def can_execute(plan: Any, config: LiveBotConfig, state: LiveBotState, current_price: float) -> bool:
    """Verifica cooldown, hysteresis y risk checks antes de despachar órdenes."""
    if plan.action != "EXECUTE":
        return True
    now = time.time()
    cooldown_ms = (
        config.get_effective_cooldown(state.last_lambda_t)
        if config.dynamic_cooldown else float(config.cooldown_ms)
    )
    if state.last_execution_time > 0 and (now - state.last_execution_time) * 1000 < cooldown_ms:
        return False
    if state.last_execution_price > 0 and current_price > 0:
        if abs(current_price - state.last_execution_price) / state.last_execution_price < config.min_price_change_pct:
            return False
    if state.current_position != 0 and abs(state.current_position) >= config.max_position_pct:
        return False
    if state.last_phi_moe < config.phi_moe_threshold or state.last_lambda_t >= config.emergency_lambda_threshold:
        return False
    return True


def get_current_mid(feed: Any) -> float:
    """Obtiene mid-price actual del order book."""
    if feed and hasattr(feed, "order_book") and feed.order_book and feed.order_book.is_initialized:
        return float(feed.order_book.mid_price or 0.0)
    return 0.0


async def log_execution(
    plan: Any,
    state: LiveBotState,
    config: LiveBotConfig,
    audit_path: Optional[Path],
    execution_history: List[ExecutionContext],
) -> None:
    """Registra la ejecución en el historial y en audit log estructurado."""
    if not config.enable_audit_log or not audit_path:
        return
    ctx = ExecutionContext(
        timestamp=time.time(),
        phi_moe=state.last_phi_moe, lambda_t=state.last_lambda_t, phi_ritmo=state.last_phi_ritmo,
        action=plan.action, side="", qty=0.0, price=0.0, order_type="",
        decision_trace=plan.envelope.metadata.get("decision_trace", {}) if plan.envelope else {},
        telemetry_hash="",
    )
    execution_history.append(ctx)
    log_file = audit_path / f"audit_{time.strftime('%Y%m%d')}.ndjson"
    try:
        import aiofiles
        async with aiofiles.open(log_file, "a") as f:
            await f.write(json.dumps({
                "timestamp": ctx.timestamp, "phi_moe": ctx.phi_moe,
                "lambda_t": ctx.lambda_t, "phi_ritmo": ctx.phi_ritmo, "action": ctx.action,
            }) + "\n")
    except Exception as e:
        logger.warning("Failed to write audit log", extra={"error": str(e)})


async def save_state(state: LiveBotState, state_path: Optional[Path | str]) -> None:
    """Guarda estado en disco de forma asíncrona."""
    if not state_path:
        return
    try:
        payload = {
            "cycle_count": state.cycle_count, "last_execution_time": state.last_execution_time,
            "last_execution_price": state.last_execution_price, "last_execution_side": state.last_execution_side,
            "last_phi_moe": state.last_phi_moe, "last_lambda_t": state.last_lambda_t,
            "last_phi_ritmo": state.last_phi_ritmo, "current_position": state.current_position,
            "total_pnl": state.total_pnl, "trades_count": state.trades_count,
            "last_error": state.last_error, "timestamp": time.time(),
        }
        import aiofiles
        async with aiofiles.open(Path(state_path), "w") as f:
            await f.write(json.dumps(payload, indent=2))
    except Exception as e:
        logger.warning("Failed to save state", extra={"error": str(e)})


async def load_state(state: LiveBotState, state_path: Optional[Path | str]) -> None:
    """Carga estado guardado en disco."""
    if not state_path or not Path(state_path).exists():
        return
    try:
        import aiofiles
        async with aiofiles.open(Path(state_path), "r") as f:
            data = json.loads(await f.read())
            for field in ("cycle_count", "trades_count"):
                setattr(state, field, data.get(field, 0))
            for field in ("last_execution_time", "last_execution_price", "last_phi_moe",
                          "last_lambda_t", "last_phi_ritmo", "current_position", "total_pnl"):
                setattr(state, field, data.get(field, 0.0))
            state.last_execution_side = data.get("last_execution_side", "")
            state.last_error = data.get("last_error")
        logger.info("State loaded", extra={"cycle": state.cycle_count})
    except Exception as e:
        logger.warning("Failed to load state", extra={"error": str(e)})


async def perform_shutdown(
    handler: Any, order_client: Any, feed: Any, account: Any,
    state: LiveBotState, config: LiveBotConfig, state_path: Optional[Path | str],
) -> None:
    """Secuencia de apagado graceful y liquidación segura."""
    if handler:
        await handler.trigger_emergency_flush("Graceful shutdown")
    if state.current_position != 0 and order_client:
        try:
            await order_client.close_position(config.symbol)
        except Exception as e:
            logger.error("Failed to close position on shutdown", extra={"error": str(e)})
    if feed:
        await feed.disconnect()
    if order_client:
        await order_client.close()
    if account and hasattr(account, "stop_auto_sync"):
        try:
            await account.stop_auto_sync()
        except Exception as e:
            logger.warning("Failed to stop account auto-sync", extra={"error": str(e)})
    await save_state(state, state_path)

