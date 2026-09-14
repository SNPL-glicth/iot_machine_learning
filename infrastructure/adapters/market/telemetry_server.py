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
                    if not command:
                        continue

                    if command == "EMERGENCY_FLUSH":
                        logger.warning("⚠️ TUI triggered EMERGENCY_FLUSH")
                        if self._command_callback:
                            try:
                                sym = data.get("symbol")
                                if asyncio.iscoroutinefunction(self._command_callback):
                                    await self._command_callback("EMERGENCY_FLUSH", symbol=sym)
                                else:
                                    res = self._command_callback("EMERGENCY_FLUSH", symbol=sym)
                                    if asyncio.iscoroutine(res):
                                        await res
                                await websocket.send(json.dumps({
                                    "type": "command_response",
                                    "command": "EMERGENCY_FLUSH",
                                    "status": "success",
                                    "message": f"Emergency flush executed successfully for {sym or 'all active symbols'}",
                                }))
                            except Exception as cmd_err:
                                logger.critical(
                                    f"CRITICAL: Failed to execute EMERGENCY_FLUSH from TUI: {cmd_err}",
                                    exc_info=True,
                                )
                                try:
                                    await websocket.send(json.dumps({
                                        "type": "command_response",
                                        "command": "EMERGENCY_FLUSH",
                                        "status": "error",
                                        "error": str(cmd_err),
                                        "message": f"Emergency flush failed: {cmd_err}",
                                    }))
                                except Exception:
                                    pass
                    elif command in ("PAUSE", "RESUME", "QUIT"):
                        logger.info(f"TUI requested {command}")
                        if self._command_callback:
                            try:
                                if asyncio.iscoroutinefunction(self._command_callback):
                                    await self._command_callback(command)
                                else:
                                    res = self._command_callback(command)
                                    if asyncio.iscoroutine(res):
                                        await res
                                await websocket.send(json.dumps({
                                    "type": "command_response",
                                    "command": command,
                                    "status": "success",
                                }))
                            except Exception as cmd_err:
                                logger.error(f"Failed to execute command {command}: {cmd_err}", exc_info=True)
                                try:
                                    await websocket.send(json.dumps({
                                        "type": "command_response",
                                        "command": command,
                                        "status": "error",
                                        "error": str(cmd_err),
                                    }))
                                except Exception:
                                    pass
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(f"Error handling TUI websocket message: {e}", exc_info=True)
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

    async def handle_emergency_flush(symbol: str | None = None) -> None:
        if not hasattr(runner, '_handler') or not runner._handler:
            logger.critical("CRITICAL: Cannot execute EMERGENCY_FLUSH: runner has no execution handler")
            raise RuntimeError("Runner has no execution handler")

        if symbol:
            symbols = [symbol.upper()]
        else:
            symbols = list(getattr(runner, "_symbols", []))
            if not symbols and hasattr(runner, "config") and hasattr(runner.config, "symbol"):
                symbols = [runner.config.symbol]
            if not symbols:
                symbols = ["SPY"]

        logger.warning("Executing emergency flush from TUI for symbols: %s", symbols)
        errors = []
        for sym in symbols:
            try:
                await runner._handler.trigger_emergency_flush("TUI Emergency Flush", symbol=sym)
            except Exception as e:
                logger.critical(
                    "CRITICAL: Failed to execute EMERGENCY_FLUSH for %s via TUI panic button: %s",
                    sym, e, exc_info=True, extra={"symbol": sym, "error": str(e)},
                )
                errors.append(f"{sym}: {e}")

        if errors:
            raise RuntimeError(f"Emergency flush failed for: {', '.join(errors)}")

    async def handle_pause() -> None:
        logger.info("Pause requested from TUI")
        if hasattr(runner, "_paused"):
            runner._paused = True

    async def handle_resume() -> None:
        logger.info("Resume requested from TUI")
        if hasattr(runner, "_paused"):
            runner._paused = False

    async def handle_quit() -> None:
        logger.info("Quit requested from TUI")
        if hasattr(runner, "shutdown"):
            asyncio.create_task(runner.shutdown())

    async def command_dispatcher(cmd: str | None = None, **kwargs) -> None:
        if not cmd:
            return
        if cmd == "EMERGENCY_FLUSH":
            await handle_emergency_flush(symbol=kwargs.get("symbol"))
        elif cmd == "PAUSE":
            await handle_pause()
        elif cmd == "RESUME":
            await handle_resume()
        elif cmd == "QUIT":
            await handle_quit()
        else:
            logger.warning("Unknown command received from TUI: %s", cmd)

    broadcaster.set_command_callback(command_dispatcher)

    await broadcaster.start()
    return broadcaster