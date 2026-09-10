"""Connection and handshake management for Alpaca WebSocket."""

from __future__ import annotations

import json
import logging
from typing import Any, List

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


async def subscribe_streams(ws: Any, symbol: str, streams: List[str]) -> None:
    """Envía suscripción para trades, quotes y bars del símbolo."""
    msg = {
        "action": "subscribe",
        "trades": [symbol] if "trades" in streams else [],
        "quotes": [symbol] if "quotes" in streams else [],
        "bars": [symbol] if "bars" in streams else [],
    }
    filtered = {k: v for k, v in msg.items() if v}
    await ws.send(json.dumps(filtered))
    logger.info("Alpaca subscription sent", extra={"symbol": symbol, "streams": streams})


def get_reconnect_delay(attempt: int, base: float = 1.0, max_d: float = 60.0) -> float:
    """Calcula retraso exponencial para reconexión con jitter/cap."""
    return min(base * (2 ** min(attempt, 6)), max_d)
