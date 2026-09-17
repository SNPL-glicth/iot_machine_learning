"""Telemetry WebSocket Server — broadcasts live bot state & handles remote config commands."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional, Set

import websockets
from iot_machine_learning.infrastructure.adapters.market.zephyr.telemetry.commands import (
    apply_bot_config_update,
    extract_bot_config_dict,
    handle_emergency_flush,
)

logger = logging.getLogger(__name__)


class TelemetryBroadcaster:
    """WebSocket server that broadcasts live bot state to React dashboard clients."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, broadcast_interval: float = 0.1):
        self.host = host
        self.port = port
        self.broadcast_interval = broadcast_interval
        self.clients: Set[Any] = set()
        self._server: Optional[websockets.Server] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._running = False
        self._latest_state: Optional[Dict[str, Any]] = None
        self._command_callback: Optional[Callable[..., Any]] = None

    def set_command_callback(self, callback: Callable[..., Any]) -> None:
        self._command_callback = callback

    def update_state(self, state: Dict[str, Any]) -> None:
        """Synchronously update the latest state (used by broadcast loop)."""
        self._latest_state = state

    async def broadcast_state(self, state: Dict[str, Any]) -> None:
        """Update state and push to clients concurrently with timeout to prevent head-of-line blocking."""
        self._latest_state = state
        if not self.clients: return
        payload = json.dumps(state, default=str)
        dead = set()
        async def _send_one(ws):
            try: await asyncio.wait_for(ws.send(payload), timeout=0.08)
            except Exception: dead.add(ws)
        await asyncio.gather(*[_send_one(c) for c in list(self.clients)], return_exceptions=True)
        if dead: self.clients -= dead


    async def register(self, websocket: Any) -> None:
        self.clients.add(websocket)
        logger.info("Dashboard client connected (%d total)", len(self.clients))
        try:
            # Send current state snapshot immediately on connect
            if self._latest_state:
                await websocket.send(json.dumps(self._latest_state, default=str))

            async for message in websocket:
                data: Any = None
                try:
                    data = json.loads(message)
                    cmd = data.get("command")
                    if not cmd or not self._command_callback:
                        continue

                    # Pass payload without the 'command' key to avoid kwarg collision
                    payload_kwargs = {k: v for k, v in data.items() if k != "command"}
                    res = self._command_callback(cmd, **payload_kwargs)
                    if asyncio.iscoroutine(res):
                        res = await res

                    await websocket.send(json.dumps({
                        "type": "command_response",
                        "command": cmd,
                        "status": "success",
                        "data": res,
                        "message": f"Command {cmd} executed successfully",
                    }, default=str))
                except Exception as e:
                    logger.error("Error processing WS command: %s", e, exc_info=True)
                    try:
                        await websocket.send(json.dumps({
                            "type": "command_response",
                            "command": data.get("command") if isinstance(data, dict) else "unknown",
                            "status": "error",
                            "error": str(e),
                        }))
                    except Exception:
                        pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info("Dashboard client disconnected (%d remaining)", len(self.clients))

    async def _broadcast_loop(self) -> None:
        """Periodic push loop for clients that connected before a state update."""
        while self._running:
            if self._latest_state and self.clients:
                payload = json.dumps(self._latest_state, default=str)
                await asyncio.gather(*[c.send(payload) for c in list(self.clients)], return_exceptions=True)
            await asyncio.sleep(self.broadcast_interval)

    async def start(self) -> None:
        self._running = True
        self._server = await websockets.serve(self.register, self.host, self.port, ping_interval=20, ping_timeout=10)
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("📡 Telemetry WS server active on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        if self._server:
            self._server.close()
            await self._server.wait_closed()


async def create_telemetry_server(runner: Any, host: str = "127.0.0.1", port: int = 8765) -> TelemetryBroadcaster:
    """Factory: create TelemetryBroadcaster and wire runner command callbacks."""
    broadcaster = TelemetryBroadcaster(host=host, port=port)

    async def command_dispatcher(cmd: str, **kwargs) -> Any:
        if cmd == "EMERGENCY_FLUSH":
            await handle_emergency_flush(runner, symbol=kwargs.get("symbol"))
            return {"flushed": True}
        elif cmd == "PAUSE":
            if hasattr(runner, "_paused"):
                runner._paused = True
            return {"paused": True}
        elif cmd == "RESUME":
            if hasattr(runner, "_paused"):
                runner._paused = False
            return {"paused": False}
        elif cmd == "GET_CONFIG":
            return extract_bot_config_dict(getattr(runner, "config", None))
        elif cmd == "UPDATE_CONFIG":
            updates = kwargs.get("config", {})
            applied = apply_bot_config_update(runner, updates)
            if hasattr(runner, "_weaviate_store"):
                asyncio.create_task(asyncio.to_thread(runner._weaviate_store.log_config_change, updates, source="dashboard_ws"))
            
            # Hot-reloading broker clients if keys changed
            rebuild_keys = {"broker", "testnet", "alpaca_api_key", "alpaca_secret_key", "alpaca_api_base_url"}
            if applied and any(k in updates for k in rebuild_keys):
                from iot_machine_learning.infrastructure.adapters.market.zephyr.execution import (
                    create_account,
                    create_order_client,
                )
                runner._order_client = create_order_client(runner.config)
                runner._account = create_account(runner.config, runner._order_client)
                if runner._handler: runner._handler.broker_client = runner._order_client
            return {"applied": applied}
        elif cmd == "QUERY_WEAVIATE_TELEMETRY":
            store = getattr(runner, "_weaviate_store", None)
            return await store.query_recent_telemetry(kwargs.get("symbol"), int(kwargs.get("limit", 50))) if store else []
        elif cmd == "QUERY_WEAVIATE_EXECUTIONS":
            store = getattr(runner, "_weaviate_store", None)
            return await store.query_recent_executions(kwargs.get("symbol"), int(kwargs.get("limit", 50))) if store else []
        elif cmd == "QUIT":
            if hasattr(runner, "shutdown"): asyncio.create_task(runner.shutdown())
            return {"shutdown": True}
        raise ValueError(f"Unknown command: {cmd}")

    broadcaster.set_command_callback(command_dispatcher)
    await broadcaster.start()
    return broadcaster