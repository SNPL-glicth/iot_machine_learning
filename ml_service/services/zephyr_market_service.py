"""Service for Zephyr Market Bot integration — discovery, telemetry & proxy."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
import websockets

from .zephyr_config_repository import ZephyrConfigRepository

logger = logging.getLogger(__name__)


class ZephyrMarketService:
    """Manages dynamic communication with the Zephyr Market Bot and configuration repository."""

    def __init__(
        self,
        ws_host: Optional[str] = None,
        ws_port: Optional[int] = None,
        repo: Optional[ZephyrConfigRepository] = None,
    ) -> None:
        self.ws_host = ws_host or os.getenv("ZEPHYR_WS_HOST", "127.0.0.1")
        self.ws_port = int(ws_port or os.getenv("ZEPHYR_WS_PORT", "8765"))
        self.repo = repo or ZephyrConfigRepository()

    @property
    def ws_url(self) -> str:
        return os.getenv("ZEPHYR_WS_URL") or f"ws://{self.ws_host}:{self.ws_port}"

    async def check_bot_alive(self, timeout: float = 0.5) -> bool:
        """Pings the Market Bot WebSocket server to verify liveness."""
        try:
            async with websockets.connect(self.ws_url, open_timeout=timeout) as ws:
                await ws.ping()
                return True
        except Exception:
            return False

    def get_connection_info(self) -> Dict[str, Any]:
        """Returns dynamic connection parameters without hardcoded paths."""
        return {
            "ws_url": self.ws_url,
            "host": self.ws_host,
            "port": self.ws_port,
            "proxy_endpoint": "/api/zephyr/ws",
            "protocol": "json_telemetry_v1",
            "reconnect_interval_ms": 2000,
        }

    def get_dynamic_accounts(self) -> List[Dict[str, Any]]:
        """Extracts available account profiles directly from Zephyr JSON files."""
        return self.repo.list_accounts()

    async def add_account_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Stores a new account profile and hot-reloads the bot if set to active."""
        profile = self.repo.create_account_profile(data)
        if data.get("set_active", False) and await self.check_bot_alive():
            await self.dispatch_command("UPDATE_CONFIG", config=self.repo.read_active_config())
        return profile

    async def switch_active_account(self, profile_id: str) -> Dict[str, Any]:
        """Activates an account profile and pushes updates to the running bot."""
        updated_fields = self.repo.activate_profile(profile_id)
        if await self.check_bot_alive():
            try:
                await self.dispatch_command("UPDATE_CONFIG", config=updated_fields)
            except Exception as e:
                logger.warning("Could not hot-reload bot on account switch: %s", e)
        return {"status": "success", "active_id": profile_id, "config": updated_fields}

    def get_active_config(self) -> Dict[str, Any]:
        """Returns sanitized public configuration from zephyr_config.json."""
        cfg = self.repo.read_active_config()
        # Sanitize sensitive fields
        sanitized = {k: v for k, v in cfg.items() if not ("secret" in k.lower() or "key" in k.lower())}
        return sanitized

    async def update_active_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Updates and persists active configuration, then hot-reloads the bot."""
        persisted = self.repo.save_active_config(updates)
        if await self.check_bot_alive():
            await self.dispatch_command("UPDATE_CONFIG", config=updates)
        return persisted

    async def get_latest_telemetry_snapshot(self, timeout: float = 1.0) -> Dict[str, Any]:
        """Fetches the immediate state snapshot broadcasted by the Market Bot."""
        try:
            async with websockets.connect(self.ws_url, open_timeout=timeout) as ws:
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                return {"online": True, "state": json.loads(raw_msg)}
        except Exception as err:
            return {
                "online": False,
                "error": str(err),
                "message": "Market bot WebSocket unreachable or offline",
            }

    async def dispatch_command(self, command: str, timeout: float = 2.5, **kwargs) -> Dict[str, Any]:
        """Sends a remote command to the Market Bot and awaits the response."""
        payload = {"command": command, **kwargs}
        try:
            async with websockets.connect(self.ws_url, open_timeout=timeout) as ws:
                await ws.send(json.dumps(payload))
                while True:
                    res_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    parsed = json.loads(res_raw)
                    if isinstance(parsed, dict) and parsed.get("type") == "command_response":
                        return parsed
        except Exception as err:
            logger.error("Failed to execute command %s on Market Bot: %s", command, err)
            return {
                "type": "command_response",
                "command": command,
                "status": "error",
                "error": str(err),
            }


_default_service: Optional[ZephyrMarketService] = None


def get_zephyr_market_service() -> ZephyrMarketService:
    global _default_service
    if _default_service is None:
        _default_service = ZephyrMarketService()
    return _default_service
