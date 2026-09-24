"""FastAPI sub-router for Zephyr Bot lifecycle control and broker account inspection."""
from __future__ import annotations

import logging
from typing import Any, Dict
from fastapi import APIRouter, Depends

from .dependencies import verify_api_key
from ..services.zephyr_bot_manager import (
    ZephyrBotManager,
    get_zephyr_bot_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/zephyr", tags=["Zephyr Bot Control"])


@router.get("/bot/status")
async def get_bot_status(
    manager: ZephyrBotManager = Depends(get_zephyr_bot_manager),
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Returns whether the Zephyr trading bot process is running, PID, and uptime."""
    return await manager.get_bot_status()


@router.post("/bot/start")
async def start_bot(
    manager: ZephyrBotManager = Depends(get_zephyr_bot_manager),
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Starts the Zephyr trading bot process with current configuration."""
    return await manager.start_bot()


@router.post("/bot/stop")
async def stop_bot(
    manager: ZephyrBotManager = Depends(get_zephyr_bot_manager),
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Gracefully terminates the Zephyr trading bot process."""
    return await manager.stop_bot()


@router.get("/broker/overview")
async def get_broker_overview(
    manager: ZephyrBotManager = Depends(get_zephyr_bot_manager),
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Returns complete broker account details: equity, cash, buying power, positions."""
    return await manager.get_broker_account_overview()
