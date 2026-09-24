"""FastAPI endpoints for Zephyr Dashboard & Market Bot communication."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .dependencies import verify_api_key
from ..services.zephyr_market_service import (
    ZephyrMarketService,
    get_zephyr_market_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/zephyr", tags=["Zephyr Market Bot"])


class CommandPayload(BaseModel):
    command: str = Field(..., description="Action name (e.g. PAUSE, RESUME, EMERGENCY_FLUSH, GET_CONFIG)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional command parameters")


class AccountCreatePayload(BaseModel):
    name: str = Field(..., description="Friendly profile name")
    broker: str = Field("alpaca", description="Broker: alpaca or binance")
    mode: str = Field("paper", description="Trading mode: paper or live")
    symbol: str = Field("SPY", description="Primary symbol")
    symbols: list[str] = Field(default_factory=list, description="Basket of symbols")
    api_key: str | None = Field(None, description="Optional API key")
    secret_key: str | None = Field(None, description="Optional Secret key")
    set_active: bool = Field(False, description="Whether to activate immediately")


@router.get("/connection")
async def get_connection_info(
    service: ZephyrMarketService = Depends(get_zephyr_market_service),
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Returns dynamic connection metadata, WS URLs and bot liveness."""
    is_alive = await service.check_bot_alive()
    info = service.get_connection_info()
    info["is_bot_online"] = is_alive
    return info


@router.get("/accounts")
async def get_accounts(
    service: ZephyrMarketService = Depends(get_zephyr_market_service),
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Returns available account environments and profiles dynamically."""
    accounts = service.get_dynamic_accounts()
    return {"accounts": accounts, "total": len(accounts)}


@router.post("/accounts")
async def create_account(
    payload: AccountCreatePayload,
    service: ZephyrMarketService = Depends(get_zephyr_market_service),
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Persists a new account profile directly into Zephyr JSON configuration."""
    return await service.add_account_profile(payload.model_dump())


@router.post("/accounts/{account_id}/activate")
async def activate_account(
    account_id: str,
    service: ZephyrMarketService = Depends(get_zephyr_market_service),
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Switches the active account in zephyr_config.json and hot-reloads the bot."""
    return await service.switch_active_account(account_id)


@router.get("/config")
async def get_configuration(
    service: ZephyrMarketService = Depends(get_zephyr_market_service),
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Returns active sanitized configuration from zephyr_config.json."""
    return service.get_active_config()


@router.put("/config")
async def update_configuration(
    updates: Dict[str, Any],
    service: ZephyrMarketService = Depends(get_zephyr_market_service),
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Persists configuration updates to zephyr_config.json and hot-reloads the bot."""
    return await service.update_active_config(updates)


@router.get("/status")
async def get_market_status(
    service: ZephyrMarketService = Depends(get_zephyr_market_service),
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Queries live telemetry and health snapshot from the Market Bot."""
    return await service.get_latest_telemetry_snapshot()


@router.post("/command")
async def post_command(
    payload: CommandPayload,
    service: ZephyrMarketService = Depends(get_zephyr_market_service),
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Dispatches a remote control command to the running Market Bot."""
    return await service.dispatch_command(payload.command, **payload.params)


@router.websocket("/ws")
async def websocket_market_bridge(
    websocket: WebSocket,
    service: ZephyrMarketService = Depends(get_zephyr_market_service),
) -> None:
    """Bi-directional WebSocket bridge proxy between Zephyr frontend and Market Bot."""
    await websocket.accept()
    import websockets

    try:
        async with websockets.connect(service.ws_url, open_timeout=2.0) as bot_ws:
            await websocket.send_text(json.dumps({
                "type": "bridge_status",
                "connected": True,
                "target": service.ws_url,
            }))

            async def bot_to_client():
                async for message in bot_ws:
                    if isinstance(message, bytes):
                        message = message.decode("utf-8")
                    await websocket.send_text(str(message))

            async def client_to_bot():
                while True:
                    data = await websocket.receive_text()
                    await bot_ws.send(data)

            done, pending = await asyncio.wait(
                [asyncio.create_task(bot_to_client()), asyncio.create_task(client_to_bot())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

    except WebSocketDisconnect:
        logger.debug("Zephyr client disconnected from WS bridge")
    except Exception as exc:
        logger.warning("Error in Zephyr Market WS bridge: %s", exc)
        try:
            await websocket.send_text(json.dumps({
                "type": "bridge_status",
                "connected": False,
                "error": str(exc),
                "message": "Market bot WebSocket unreachable",
            }))
            await websocket.close()
        except Exception:
            pass
