"""Telemetry builder and periodic health-checks for LiveBotRunner."""

from __future__ import annotations

import logging
import time
from typing import Any, Deque, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


async def build_telemetry_state(
    state: Any,
    config: Any,
    feed: Any,
    order_client: Any,
    account: Any,
    latency_samples: Deque[float],
) -> Dict[str, Any]:
    """Construye el estado de telemetría para el TUI y clientes WebSocket."""
    equity, cash, buying_power = 0.0, 0.0, 0.0
    positions: Dict[str, Any] = {}
    orders = []

    if account:
        try:
            equity = await account.get_equity()
            cash = await account.get_cash() if hasattr(account, "get_cash") else 0.0
            buying_power = await account.get_buying_power() if hasattr(account, "get_buying_power") else 0.0
            if hasattr(account, "get_all_positions"):
                pos_dict = await account.get_all_positions()
                positions = {k: v for k, v in pos_dict.items()} if pos_dict else {}
            elif hasattr(account, "get_position"):
                pos = await account.get_position(config.symbol)
                if pos:
                    positions[config.symbol] = pos
            if hasattr(order_client, "get_orders"):
                orders_list = await order_client.get_orders(status="open", limit=50)
                orders = [{"id": o.id, "symbol": o.symbol, "side": o.side, "type": o.order_type,
                           "qty": o.qty, "price": o.price, "status": o.status} for o in orders_list]
        except Exception as e:
            logger.debug(f"Error fetching account data for telemetry: {e}")

    best_bid, best_ask, bid_vol, ask_vol, obi, microprice = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if feed:
        if hasattr(feed, "order_book") and feed.order_book:
            best_bid = feed.order_book.best_bid or 0.0
            best_ask = feed.order_book.best_ask or 0.0
            if feed.order_book.metrics:
                bid_vol = feed.order_book.metrics.bid_volume
                ask_vol = feed.order_book.metrics.ask_volume
                obi = feed.order_book.metrics.volume_imbalance
                microprice = feed.order_book.metrics.microprice
        elif getattr(feed, "best_bid", None) is not None and getattr(feed, "best_ask", None) is not None:
            best_bid = float(feed.best_bid or 0.0)
            best_ask = float(feed.best_ask or 0.0)
            q = getattr(feed, "_latest_quote", None)
            if q:
                bid_vol = float(getattr(q, "bid_size", 0.0))
                ask_vol = float(getattr(q, "ask_size", 0.0))
                tot = bid_vol + ask_vol
                obi = (bid_vol - ask_vol) / tot if tot > 0 else 0.0
                microprice = (best_bid * ask_vol + best_ask * bid_vol) / tot if tot > 0 else (best_bid + best_ask) / 2.0

    mode = ("PAPER" if config.is_paper_trading else "LIVE") if config.broker == "alpaca" else ("TESTNET" if config.testnet else "MAINNET")
    market_open, next_open, next_close = None, None, None
    if hasattr(order_client, "get_clock"):
        try:
            clock = await order_client.get_clock()
            market_open = bool(clock.get("is_open"))
            next_open, next_close = clock.get("next_open"), clock.get("next_close")
        except Exception as e:
            logger.debug(f"Error fetching market clock for telemetry: {e}")

    return {
        "timestamp": time.time(), "symbol": config.symbol, "mode": mode, "broker": config.broker,
        "latency_p50_ms": float(np.percentile(latency_samples, 50)) if latency_samples else 0.0,
        "phi_moe": state.last_phi_moe, "lambda_t": state.last_lambda_t, "phi_ritmo": state.last_phi_ritmo,
        "best_bid": best_bid, "best_ask": best_ask, "bid_vol": bid_vol, "ask_vol": ask_vol,
        "obi": obi, "microprice": microprice, "experts": getattr(state, "last_expert_votes", []),
        "position_qty": state.current_position, "entry_price": state.last_execution_price,
        "pnl_usd": state.total_pnl, "pnl_pct": 0.0,
        "last_action": getattr(state, "last_action", "HOLD"),
        "last_reason": getattr(state, "last_reason", ""),
        "decision_rationale": {
            "action": getattr(state, "last_action", "HOLD"), "reason": getattr(state, "last_reason", ""),
            "phi_moe": state.last_phi_moe, "lambda_t": state.last_lambda_t, "phi_ritmo": state.last_phi_ritmo,
            "expert_votes": {e["name"]: e.get("vote", 0.0) for e in getattr(state, "last_expert_votes", [])},
            "regime": "stable", "risk_checks": [], "timestamp": int(time.time() * 1000),
        },
        "equity": equity, "cash": cash, "buying_power": buying_power,
        "positions": positions, "orders": orders,
        "feed_connected": feed.is_connected if feed else False,
        "feed_state": str(feed.state) if feed else "DISCONNECTED",
        "market_open": market_open, "next_open_et": next_open, "next_close_et": next_close,
        "session_note": "RTH 9:30-16:00 ET; extendida 04:00-20:00 ET",
    }


async def perform_health_check(
    state: Any,
    config: Any,
    feed: Any,
    latency_samples: Deque[float],
    running: bool,
    callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Ejecuta un health check completo y emite alertas pertinentes."""
    p50 = float(np.percentile(latency_samples, 50)) if latency_samples else 0.0
    p99 = float(np.percentile(latency_samples, 99)) if latency_samples else 0.0
    health = {
        "timestamp": time.time(), "symbol": config.symbol, "running": running,
        "feed_connected": feed.is_connected if feed else False,
        "feed_state": feed.state.value if (feed and hasattr(feed.state, "value")) else str(getattr(feed, "state", "none")),
        "order_book_initialized": feed.order_book.is_initialized if (feed and hasattr(feed, "order_book") and feed.order_book) else False,
        "order_book_metrics": feed.order_book.metrics.to_dict() if (feed and hasattr(feed, "order_book") and feed.order_book and feed.order_book.metrics) else None,
        "engine_phi_moe": state.last_phi_moe, "engine_lambda_t": state.last_lambda_t,
        "engine_phi_ritmo": state.last_phi_ritmo, "position": state.current_position,
        "trades_count": state.trades_count, "latency_p50_ms": p50, "latency_p99_ms": p99,
        "errors": state.last_error,
    }
    logger.info("Health check", extra=health)
    if latency_samples and p99 > config.max_latency_ms:
        logger.warning("High latency detected", extra={"p99_latency_ms": p99})
    if feed and not feed.is_connected:
        logger.warning("Feed disconnected", extra={"symbol": config.symbol})
    if callback:
        try:
            callback(health)
        except Exception as e:
            logger.error("Error in health check callback", extra={"error": str(e)})
    return health


def format_status_line(state: Any, feed: Any, start_time: float) -> str:
    """Genera línea de estado compacta para CLI/logs."""
    uptime = time.time() - start_time
    hh, rem = divmod(int(uptime), 3600)
    mm, ss = divmod(rem, 60)
    feed_conn = "UP" if (feed and feed.is_connected) else "DOWN"
    ob_init = "INIT" if (feed and hasattr(feed, "order_book") and feed.order_book and feed.order_book.is_initialized) else "SYNCING"
    return (
        f"up={hh:02d}:{mm:02d}:{ss:02d} | "
        f"phi={state.last_phi_moe:.3f} | "
        f"lambda={state.last_lambda_t:.3f} | "
        f"phi_r={state.last_phi_ritmo:.3f} | "
        f"pos={state.current_position:.6f} | "
        f"trades={state.trades_count} | "
        f"feed={feed_conn} | "
        f"ob={ob_init}"
    )
