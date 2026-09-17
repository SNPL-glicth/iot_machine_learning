"""Telemetry builder and periodic health-checks for LiveBotRunner."""

from __future__ import annotations

import logging
import time
from typing import Any, Deque, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


_REST_CACHE: Dict[str, Any] = {"last_fetch": 0.0, "equity": 0.0, "cash": 0.0, "buying_power": 0.0, "positions": {}, "orders": [], "clock": (None, None, None)}


async def build_telemetry_state(
    state: Any, config: Any, feed: Any, order_client: Any, account: Any, latency_samples: Deque[float],
) -> Dict[str, Any]:
    """Construye el estado de telemetría para el TUI y clientes WebSocket con TTL caching en REST."""
    now = time.time()
    if account and (now - _REST_CACHE["last_fetch"] >= 2.5):
        try:
            _REST_CACHE["equity"] = await account.get_equity()
            _REST_CACHE["cash"] = await account.get_cash() if hasattr(account, "get_cash") else 0.0
            _REST_CACHE["buying_power"] = await account.get_buying_power() if hasattr(account, "get_buying_power") else 0.0
            if hasattr(account, "get_all_positions"):
                pos_dict = await account.get_all_positions()
                _REST_CACHE["positions"] = {k: v for k, v in pos_dict.items()} if pos_dict else {}
            elif hasattr(account, "get_position"):
                pos = await account.get_position(config.symbol)
                _REST_CACHE["positions"] = {config.symbol: pos} if pos else {}
            if hasattr(order_client, "get_orders"):
                orders_list = await order_client.get_orders(status="open", limit=50)
                _REST_CACHE["orders"] = [{"id": o.id, "symbol": o.symbol, "side": o.side, "type": o.order_type,
                                          "qty": o.qty, "price": o.price, "status": o.status} for o in orders_list]
            if hasattr(order_client, "get_clock"):
                clock = await order_client.get_clock()
                _REST_CACHE["clock"] = (bool(clock.get("is_open")), clock.get("next_open"), clock.get("next_close"))
            _REST_CACHE["last_fetch"] = now
        except Exception as e:
            logger.debug(f"Error updating REST cache for telemetry: {e}")

    equity, cash, buying_power = _REST_CACHE["equity"], _REST_CACHE["cash"], _REST_CACHE["buying_power"]
    positions, orders = _REST_CACHE["positions"], _REST_CACHE["orders"]

    best_bid, best_ask, bid_vol, ask_vol, obi, microprice = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if feed:
        if hasattr(feed, "order_book") and feed.order_book:
            best_bid, best_ask = feed.order_book.best_bid or 0.0, feed.order_book.best_ask or 0.0
            if feed.order_book.metrics:
                bid_vol = feed.order_book.metrics.bid_volume
                ask_vol = feed.order_book.metrics.ask_volume
                obi = feed.order_book.metrics.volume_imbalance
                microprice = feed.order_book.metrics.microprice
        elif getattr(feed, "best_bid", None) is not None and getattr(feed, "best_ask", None) is not None:
            best_bid, best_ask = float(feed.best_bid or 0.0), float(feed.best_ask or 0.0)
            q = getattr(feed, "_latest_quote", None)
            if q:
                bid_vol, ask_vol = float(getattr(q, "bid_size", 0.0)), float(getattr(q, "ask_size", 0.0))
                tot = bid_vol + ask_vol
                obi = (bid_vol - ask_vol) / tot if tot > 0 else 0.0
                microprice = (best_bid * ask_vol + best_ask * bid_vol) / tot if tot > 0 else (best_bid + best_ask) / 2.0

    mode = ("PAPER" if config.is_paper_trading else "LIVE") if config.broker == "alpaca" else ("TESTNET" if config.testnet else "MAINNET")
    market_open, next_open, next_close = _REST_CACHE["clock"]


    # Normalize positions dict → list expected by React dashboard
    positions_list: list = []
    if isinstance(positions, dict):
        for sym, pos_data in positions.items():
            if isinstance(pos_data, dict):
                entry = dict(pos_data)
                entry.setdefault("symbol", sym)
                positions_list.append(entry)
            elif hasattr(pos_data, "__dict__"):
                entry = {k: v for k, v in vars(pos_data).items()}
                entry.setdefault("symbol", sym)
                positions_list.append(entry)
    elif isinstance(positions, list):
        positions_list = positions

    last_action = getattr(state, "last_action", "HOLD") or "HOLD"
    last_reason = getattr(state, "last_reason", "") or ""
    expert_votes = getattr(state, "last_expert_votes", []) or []

    return {
        "timestamp": time.time(),
        "symbol": config.symbol,
        "mode": mode,
        "broker": config.broker,
        "latency_p50_ms": float(np.percentile(latency_samples, 50)) if latency_samples else 0.0,
        "phi_moe": float(state.last_phi_moe or 0.0),
        "lambda_t": float(state.last_lambda_t or 0.0),
        "phi_ritmo": float(state.last_phi_ritmo or 0.0),
        "best_bid": float(best_bid or 0.0),
        "best_ask": float(best_ask or 0.0),
        "bid_vol": float(bid_vol or 0.0),
        "ask_vol": float(ask_vol or 0.0),
        "obi": float(obi or 0.0),
        "microprice": float(microprice or 0.0),
        "experts": expert_votes,
        "equity": float(equity or 0.0),
        "cash": float(cash or 0.0),
        "buying_power": float(buying_power or 0.0),
        "positions": positions_list,
        "orders": orders,
        "last_action": last_action,
        "last_reason": last_reason,
        "decision_rationale": last_reason,
        "feed_connected": bool(feed.is_connected) if feed else False,
        "feed_state": str(feed.state) if feed else "DISCONNECTED",
        "market_open": market_open,
        "next_open_et": next_open,
        "next_close_et": next_close,
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
