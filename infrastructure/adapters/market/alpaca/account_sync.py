"""Synchronization helpers for Alpaca Account state."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from iot_machine_learning.infrastructure.adapters.market.alpaca.account_models import Position

logger = logging.getLogger(__name__)


def parse_position(pos: Dict[str, Any]) -> Position:
    """Parsea el diccionario de posición devuelto por Alpaca a una entidad Position."""
    sym = pos["symbol"]
    return Position(
        symbol=sym,
        side=pos["side"],
        qty=float(pos["qty"]),
        avg_entry_price=float(pos["avg_entry_price"]),
        market_value=float(pos["market_value"]),
        cost_basis=float(pos["cost_basis"]),
        unrealized_pl=float(pos["unrealized_pl"]),
        unrealized_plpc=float(pos["unrealized_plpc"]),
        current_price=float(pos["current_price"]),
        lastday_price=float(pos["lastday_price"]),
        change_today=float(pos["change_today"]),
        asset_id=pos["asset_id"],
        asset_class=pos["asset_class"],
        exchange=pos["exchange"],
    )


async def fetch_account_and_positions(client: Any) -> Tuple[Dict[str, Any], Dict[str, Position]]:
    """Consulta la API de Alpaca y devuelve datos normalizados de cuenta y posiciones."""
    account = await client.get_account()
    account_fields: Dict[str, Any] = {
        "equity": float(account.get("equity", 0)),
        "cash": float(account.get("cash", 0)),
        "buying_power": float(account.get("buying_power", 0)),
        "portfolio_value": float(account.get("portfolio_value", 0)),
        "pattern_day_trader": account.get("pattern_day_trader", False),
        "trading_blocked": account.get("trading_blocked", False),
        "transfers_blocked": account.get("transfers_blocked", False),
        "account_blocked": account.get("account_blocked", False),
    }

    positions_data = await client.get_positions()
    positions: Dict[str, Position] = {
        pos["symbol"]: parse_position(pos) for pos in positions_data
    }

    return account_fields, positions
