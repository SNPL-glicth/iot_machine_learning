"""Execution logic, risk checks, auditing and persistence for LiveBotRunner."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.live_runner_models import (
    ExecutionContext,
    LiveBotState,
)

logger = logging.getLogger(__name__)


def can_execute(
    plan: Any, config: LiveBotConfig, state: LiveBotState, current_price: float,
    symbol: str | None = None, risk_mgr: Any = None, macro_velocity: float = 0.0,
) -> bool:
    """Verifica cooldown, hysteresis, sesión de mercado, disyuntor y risk checks antes de despachar órdenes."""
    if plan.action != "EXECUTE":
        return True
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
        if getattr(traj, "terminal_state", None) is not None and hasattr(traj.terminal_state, "state_vector"):
            vec = getattr(traj.terminal_state, "state_vector", None)
            if vec is not None and len(vec) > 0 and current_price > 0:
                side = "buy" if float(vec[0]) > current_price else "sell"
        if not side and hasattr(traj, "side"):
            side = str(getattr(traj, "side", ""))
    if risk_mgr:
        if not risk_mgr.check_macro_velocity(sym, side, macro_velocity)[0]:
            return False
        if not risk_mgr.check_correlation_guardrail(sym, side, state.positions)[0]:
            return False
    if symbol:
        if state.get_position(symbol) != 0:
            return False
        if state.active_positions_count >= getattr(config, "max_concurrent_positions", 2):
            return False
    elif state.active_positions_count > 0:
        return False
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
    if state.last_phi_moe < config.phi_moe_threshold or state.last_lambda_t >= config.emergency_lambda_threshold:
        return False
    return True


def get_current_mid(feed: Any, symbol: str | None = None) -> float:
    """Obtiene mid-price actual del order book o feed para un símbolo."""
    if symbol and hasattr(feed, "get_mid_price"):
        p = feed.get_mid_price(symbol)
        if p is not None:
            return float(p)
    if feed and hasattr(feed, "order_book") and feed.order_book and getattr(feed.order_book, "is_initialized", False):
        return float(feed.order_book.mid_price or 0.0)
    if feed and hasattr(feed, "mid_price") and feed.mid_price is not None:
        return float(feed.mid_price)
    return 0.0


async def log_execution(
    plan: Any,
    state: LiveBotState,
    config: LiveBotConfig,
    audit_path: Path | None,
    execution_history: list[ExecutionContext],
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
        audit_path.mkdir(parents=True, exist_ok=True)
        import aiofiles
        async with aiofiles.open(log_file, "a") as f:
            await f.write(json.dumps({
                "timestamp": float(ctx.timestamp),
                "phi_moe": float(ctx.phi_moe),
                "lambda_t": float(ctx.lambda_t),
                "phi_ritmo": float(ctx.phi_ritmo),
                "action": str(ctx.action),
            }) + "\n")
    except Exception as e:
        logger.warning("Failed to write audit log: %s", e)


async def save_state(state: LiveBotState, state_path: Path | str | None) -> None:
    """Guarda estado en disco de forma asíncrona."""
    if not state_path:
        return
    try:
        payload = {
            "cycle_count": int(state.cycle_count),
            "last_execution_time": float(state.last_execution_time),
            "last_execution_price": float(state.last_execution_price),
            "last_execution_side": str(state.last_execution_side or ""),
            "last_phi_moe": float(state.last_phi_moe),
            "last_lambda_t": float(state.last_lambda_t),
            "last_phi_ritmo": float(state.last_phi_ritmo),
            "current_position": float(state.current_position),
            "positions": {k: float(v) for k, v in state.positions.items()},
            "total_pnl": float(state.total_pnl),
            "trades_count": int(state.trades_count),
            "consecutive_losses": {k: int(v) for k, v in state.consecutive_losses.items()},
            "streak_cooling_until": {k: float(v) for k, v in state.streak_cooling_until.items()},
            "portfolio_circuit_breaker_tripped": bool(state.portfolio_circuit_breaker_tripped),
            "last_error": str(state.last_error) if state.last_error else None,
            "timestamp": time.time(),
        }
        p = Path(state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        import aiofiles
        async with aiofiles.open(p, "w") as f:
            await f.write(json.dumps(payload, indent=2))
    except Exception as e:
        logger.warning("Failed to save state: %s", e)


async def load_state(state: LiveBotState, state_path: Path | str | None) -> None:
    """Carga estado guardado en disco."""
    if not state_path or not Path(state_path).exists():
        return
    try:
        import aiofiles
        async with aiofiles.open(Path(state_path)) as f:
            data = json.loads(await f.read())
            for field in ("cycle_count", "trades_count"):
                setattr(state, field, data.get(field, 0))
            for field in ("last_execution_time", "last_execution_price", "last_phi_moe",
                          "last_lambda_t", "last_phi_ritmo", "current_position", "total_pnl"):
                setattr(state, field, data.get(field, 0.0))
            state.positions = data.get("positions", {})
            state.consecutive_losses = data.get("consecutive_losses", {})
            state.streak_cooling_until = data.get("streak_cooling_until", {})
            state.portfolio_circuit_breaker_tripped = bool(data.get("portfolio_circuit_breaker_tripped", False))
            state.last_execution_side = data.get("last_execution_side", "")
            state.last_error = data.get("last_error")
        logger.info("State loaded", extra={"cycle": state.cycle_count})
    except Exception as e:
        logger.warning("Failed to load state: %s", e)


async def perform_shutdown(
    handler: Any, order_client: Any, feed: Any, account: Any,
    state: LiveBotState, config: LiveBotConfig, state_path: Path | str | None,
) -> None:
    """Secuencia de apagado graceful y liquidación segura."""
    if handler:
        await handler.trigger_emergency_flush("Graceful shutdown")
    if order_client and hasattr(order_client, "close_position"):
        open_syms = [s for s, q in state.positions.items() if q != 0.0]
        if not open_syms and state.get_position(config.symbol) != 0.0:
            open_syms = [config.symbol]
        for s in open_syms:
            try:
                await order_client.close_position(s)
            except Exception as e:
                logger.error("Failed to close position for %s on shutdown: %s", s, e)
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

