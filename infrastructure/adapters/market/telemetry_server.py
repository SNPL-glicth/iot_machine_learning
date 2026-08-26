"""Telemetry WebSocket Server — broadcasts live bot state to TypeScript TUI."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional, Set

import websockets

logger = logging.getLogger(__name__)


class TelemetryBroadcaster:
    """
    WebSocket server that broadcasts live bot state to TypeScript TUI clients.
    
    Protocol:
    - Clients connect to ws://127.0.0.1:8765
    - Server pushes TelemetryFrame JSON at ~10Hz
    - Clients can send commands: EMERGENCY_FLUSH, PAUSE, RESUME
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        broadcast_interval: float = 0.1,  # 10 Hz
    ):
        self.host = host
        self.port = port
        self.broadcast_interval = broadcast_interval
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self._server: Optional[websockets.Server] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._running = False
        self._latest_state: Optional[Dict[str, Any]] = None
        self._command_callback: Optional[callable] = None

    def set_command_callback(self, callback: callable) -> None:
        """Set callback for handling commands from TUI (e.g., EMERGENCY_FLUSH)."""
        self._command_callback = callback

    def update_state(self, state: Dict[str, Any]) -> None:
        """Update latest state to be broadcasted."""
        self._latest_state = state

    async def register(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """Register new TUI client."""
        self.clients.add(websocket)
        logger.info(f"TUI client connected ({len(self.clients)} total)")

        try:
            # Send current state immediately
            if self._latest_state:
                await websocket.send(json.dumps(self._latest_state))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    command = data.get("command")
                    if command == "EMERGENCY_FLUSH" and self._command_callback:
                        logger.warning("⚠️ TUI triggered EMERGENCY_FLUSH")
                        await self._command_callback()
                    elif command == "PAUSE":
                        logger.info("TUI requested PAUSE")
                        if self._command_callback:
                            await self._command_callback("PAUSE")
                    elif command == "RESUME":
                        logger.info("TUI requested RESUME")
                        if self._command_callback:
                            await self._command_callback("RESUME")
                    elif command == "QUIT":
                        logger.info("TUI requested QUIT")
                        if self._command_callback:
                            await self._command_callback("QUIT")
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"TUI client disconnected ({len(self.clients)} remaining)")

    async def _broadcast_loop(self) -> None:
        """Periodically broadcast latest state to all connected clients."""
        while self._running:
            if self._latest_state and self.clients:
                payload = json.dumps(self._latest_state)
                # Send to all clients concurrently
                await asyncio.gather(
                    *[client.send(payload) for client in self.clients],
                    return_exceptions=True
                )
            await asyncio.sleep(self.broadcast_interval)

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._running = True
        self._server = await websockets.serve(
            self.register,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
        )
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info(f"📡 Telemetry Server WS active on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the server gracefully."""
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
        logger.info("Telemetry server stopped")

    async def broadcast_state(self, state: Dict[str, Any]) -> None:
        """Manually broadcast a state update (for immediate pushes)."""
        self._latest_state = state
        if self.clients:
            payload = json.dumps(state)
            await asyncio.gather(
                *[client.send(json.dumps(state)) for client in self.clients],
                return_exceptions=True
            )


# For backward compatibility
import json


async def create_telemetry_server(
    runner,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> TelemetryBroadcaster:
    """Factory to create and configure telemetry server with runner callbacks."""
    broadcaster = TelemetryBroadcaster(host=host, port=port)

    async def handle_emergency_flush():
        if hasattr(runner, '_handler') and runner._handler:
            runner._handler.trigger_emergency_flush("TUI Emergency Flush")

    async def handle_pause():
        logger.info("Pause requested from TUI")
        # Could set a pause flag on runner

    async def handle_resume():
        logger.info("Resume requested from TUI")

    async def handle_quit():
        logger.info("Quit requested from TUI")
        # Could trigger graceful shutdown

    broadcaster.set_command_callback(lambda cmd=None: {
        "EMERGENCY_FLUSH": handle_emergency_flush,
        "PAUSE": handle_pause,
        "RESUME": handle_resume,
        "QUIT": handle_quit,
    }.get(cmd, lambda: None)() if cmd else None)

    await broadcaster.start()
    return broadcaster