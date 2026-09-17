"""Market session checks, position synchronization and health check routines."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable

from iot_machine_learning.infrastructure.adapters.market.zephyr.config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.zephyr.models import LiveBotState
from iot_machine_learning.infrastructure.adapters.market.zephyr.telemetry.builder import perform_health_check

logger = logging.getLogger(__name__)


async def check_market_session(
    state: LiveBotState,
    config: LiveBotConfig,
    order_client: Any,
    handler: Any,
    portfolio_risk_mgr: Any,
    symbols: list[str],
    get_equity_fn: Callable[[], Any],
) -> None:
    """Monitorea el reloj de mercado de Alpaca (cierre RTH 16:00 ET)."""
    if not order_client or not hasattr(order_client, "get_clock"):
        return
    try:
        clock = await order_client.get_clock()
        is_open = bool(clock.get("is_open", False))
        next_close_str = clock.get("next_close")
        if not is_open:
            if getattr(config, "enforce_market_hours", False):
                if not state.market_closed:
                    state.market_closed = True
                    state.market_closing_soon = True
                    logger.info("Market session is CLOSED (clock.is_open=False). Inhibiting entries.")
                    if handler:
                        for sym in symbols:
                            await handler.trigger_emergency_flush("Market Close", symbol=sym)
            return

        if state.market_closed or state.market_closing_soon:
            state.market_closed = False
            state.market_closing_soon = False
            logger.info("Market session is OPEN (clock.is_open=True). Resuming entry orders.")
            if portfolio_risk_mgr:
                try:
                    portfolio_risk_mgr.reset_day(await get_equity_fn())
                except Exception as e:
                    logger.warning("Failed to reset daily risk baseline: %s", e)

        if next_close_str and getattr(config, "enforce_market_hours", False):
            close_dt = datetime.fromisoformat(next_close_str)
            now_dt = datetime.fromisoformat(clock.get("timestamp", datetime.now().isoformat()))
            sec_to_close = (close_dt - now_dt).total_seconds()
            if sec_to_close <= 300 and not state.market_closing_soon:
                state.market_closing_soon = True
                logger.warning("Market closing in %.1f minutes (~15:55 ET). Inhibiting entries.", sec_to_close / 60.0)
            elif sec_to_close <= 10 and not state.market_closed:
                state.market_closed = True
                state.market_closing_soon = True
                logger.warning("Market session closing imminently (16:00 ET). Flattening positions.")
                if handler:
                    for sym in symbols:
                        await handler.trigger_emergency_flush("End of Day Close", symbol=sym)
    except Exception as e:
        logger.debug("Error checking market session: %s", e)


async def sync_and_check_health(
    state: LiveBotState,
    config: LiveBotConfig,
    account: Any,
    handler: Any,
    portfolio_risk_mgr: Any,
    symbols: list[str],
    get_equity_fn: Callable[[], Any],
    feed: Any,
    latency_samples: Any,
    running: bool,
    last_health_check: float,
) -> float:
    """Verifica estado de cuenta, sincroniza posiciones y chequea riesgo de cartera."""
    if account and hasattr(account, "is_tradeable"):
        try:
            tradeable = await account.is_tradeable()
            state.account_blocked = not tradeable
            if not tradeable:
                logger.warning("Account health check: account is NOT tradeable. Entry orders vetoed.")
        except Exception as e:
            logger.warning("Failed to check account tradeable status: %s", e)
            state.account_blocked = True

    if account and hasattr(account, "get_position"):
        try:
            for sym in symbols:
                pos = await account.get_position(sym)
                old_pos = state.get_position(sym)
                state.set_position(sym, pos)
                if old_pos != 0 and pos == 0:
                    state.mark_closing(sym)
                    state.mark_close_confirmed(sym)
            if hasattr(account, "get_unrealized_pl"):
                state.total_pnl = await account.get_unrealized_pl(config.symbol)
        except Exception as e:
            logger.debug("Position sync error: %s", e)

    if portfolio_risk_mgr and (getattr(config, "enforce_portfolio_profit_lock", True) or getattr(config, "max_daily_loss_usd", 0.0) > 0):
        try:
            current_eq = await get_equity_fn()
            tripped, reason = portfolio_risk_mgr.update_equity(current_eq)
            if tripped and not state.portfolio_circuit_breaker_tripped:
                state.portfolio_circuit_breaker_tripped = True
                logger.critical("PORTFOLIO CIRCUIT BREAKER TRIPPED: %s", reason)
                if handler:
                    for s in symbols:
                        await handler.trigger_emergency_flush(reason, symbol=s)
        except Exception as e:
            logger.debug("Portfolio risk evaluation error: %s", e)

    now = time.time()
    if now - last_health_check >= config.health_check_interval_sec:
        await perform_health_check(state, config, feed, latency_samples, running)
        return now
    return last_health_check
