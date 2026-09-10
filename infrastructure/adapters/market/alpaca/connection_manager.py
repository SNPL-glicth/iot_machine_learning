"""Connection and handshake management for Alpaca WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from iot_machine_learning.domain.entities.market.observations import MarketObservation
from iot_machine_learning.infrastructure.adapters.market.alpaca.feed_models import FeedStats
from iot_machine_learning.infrastructure.adapters.market.alpaca.message_parser import (
    parse_raw_message,
)

logger = logging.getLogger(__name__)


async def connect_and_auth(ws_url: str, api_key: str, api_secret: str, symbol: str) -> Any:
    """Establece conexión WebSocket con Alpaca y realiza el handshake de autenticación."""
    import websockets

    ws = await websockets.connect(
        ws_url,
        ping_interval=None,
        ping_timeout=None,
        close_timeout=10,
    )
    auth_msg = {"action": "auth", "key": api_key, "secret": api_secret}
    await ws.send(json.dumps(auth_msg))

    conn_resp = await ws.recv()
    conn_data = json.loads(conn_resp)
    if not (isinstance(conn_data, list) and conn_data and conn_data[0].get("T") == "success" and conn_data[0].get("msg") == "connected"):
        raise RuntimeError(f"Connection failed: {conn_data}")

    auth_resp = await ws.recv()
    auth_data = json.loads(auth_resp)
    if not (isinstance(auth_data, list) and auth_data and auth_data[0].get("T") == "success" and auth_data[0].get("msg") == "authenticated"):
        raise RuntimeError(f"Authentication failed: {auth_data}")

    logger.info("Alpaca WebSocket authenticated", extra={"symbol": symbol})
    return ws


async def subscribe_streams(ws: Any, symbol: str | list[str], streams: list[str]) -> None:
    """Envía suscripción para trades, quotes y bars de los símbolos."""
    sym_list = [symbol] if isinstance(symbol, str) else list(symbol)
    msg = {
        "action": "subscribe",
        "trades": sym_list if "trades" in streams else [],
        "quotes": sym_list if "quotes" in streams else [],
        "bars": sym_list if "bars" in streams else [],
    }
    filtered = {k: v for k, v in msg.items() if v}
    await ws.send(json.dumps(filtered))
    logger.info("Alpaca subscription sent", extra={"symbols": sym_list, "streams": streams})


def get_reconnect_delay(attempt: int, base: float = 1.0, max_d: float = 60.0) -> float:
    """Calcula retraso exponencial para reconexión con jitter/cap."""
    delay: float = base * (2.0 ** min(attempt, 6))
    return delay if delay < max_d else max_d


async def run_ping_loop(ws_getter: Callable[[], Any], is_active: Callable[[], bool], reconnect_cb: Callable[[], Any], interval: float = 20.0) -> None:
    """Mantiene el WebSocket vivo mediante pings periódicos."""
    while is_active():
        await asyncio.sleep(interval)
        ws = ws_getter()
        if ws:
            try:
                await ws.ping()
            except Exception as e:
                logger.warning("Ping failed", extra={"error": str(e)})
                await reconnect_cb()


async def run_health_loop(is_active: Callable[[], bool], stats_fn: Callable[[], dict], on_metrics: Callable[[dict], None] | None, interval: float = 10.0) -> None:
    """Monitorea salud y emite estadísticas periódicamente."""
    while is_active():
        await asyncio.sleep(interval)
        if not is_active():
            break
        if callable(on_metrics):
            try:
                on_metrics(stats_fn())
            except Exception as e:
                logger.error("Metrics callback error", extra={"error": str(e)})


async def perform_reconnect(
    ws_url: str, api_key: str, api_secret: str, symbol: str, symbols: list[str], streams: list[str], attempt: int
) -> Any:
    """Ejecuta retardo exponencial y reconexión completa con re-suscripción."""
    delay = get_reconnect_delay(attempt)
    logger.info(f"Reconnecting in {delay:.1f}s...", extra={"attempt": attempt})
    await asyncio.sleep(delay)
    ws = await connect_and_auth(ws_url, api_key, api_secret, symbol)
    await subscribe_streams(ws, symbols, streams)
    return ws


async def run_emit_loop(
    ws_getter: Callable[[], Any],
    is_running: Callable[[], bool],
    reconnect_cb: Callable[[], Any],
    on_observation: Callable[[MarketObservation], None] | None,
    obs_queue: asyncio.Queue[MarketObservation],
    symbol: str,
    bar_interval: str,
    stats: FeedStats,
    on_quote_cb: Callable[[Any], None] | None = None,
) -> None:
    """Loop principal de consumo de mensajes crudos y despacho a la cola de observaciones."""
    try:
        while is_running():
            ws = ws_getter()
            if not ws:
                await asyncio.sleep(0.05)
                continue
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                stats.events_received += 1
                for obs in parse_raw_message(msg, symbol, bar_interval, time.time(), stats):
                    if hasattr(obs, "bid") and hasattr(obs, "ask") and on_quote_cb:
                        on_quote_cb(obs)
                    stats.events_emitted += 1
                    stats.last_emitted_time = time.time()
                    lat_ms = (time.time() - obs.timestamp) * 1000
                    stats.avg_latency_ms = stats.avg_latency_ms * 0.99 + lat_ms * 0.01
                    stats.max_latency_ms = max(stats.max_latency_ms, lat_ms)
                    if obs_queue.full():
                        try:
                            obs_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    obs_queue.put_nowait(obs)
                    if on_observation:
                        try:
                            on_observation(obs)
                        except Exception as e:
                            logger.error("Callback error", extra={"error": str(e)})
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Emit error", extra={"symbol": symbol, "error": str(e)})
                if is_running():
                    await reconnect_cb()
    except asyncio.CancelledError:
        pass


def start_feed_tasks(
    ws_getter: Callable[[], Any],
    is_running: Callable[[], bool],
    reconnect_cb: Callable[[], Any],
    on_obs: Callable[[MarketObservation], None] | None,
    obs_queue: asyncio.Queue[MarketObservation],
    symbol: str,
    bar_interval: str,
    stats: FeedStats,
    on_quote: Callable[[Any], None] | None,
    on_metrics: Callable[[dict], None] | None,
) -> tuple[asyncio.Task, asyncio.Task, asyncio.Task]:
    """Inicia tareas de fondo del feed (emit, ping, health)."""
    emit_task = asyncio.create_task(
        run_emit_loop(ws_getter, is_running, reconnect_cb, on_obs, obs_queue, symbol, bar_interval, stats, on_quote)
    )
    ping_task = asyncio.create_task(run_ping_loop(ws_getter, is_running, reconnect_cb))
    health_task = asyncio.create_task(run_health_loop(is_running, stats.to_dict, on_metrics))
    return emit_task, ping_task, health_task

