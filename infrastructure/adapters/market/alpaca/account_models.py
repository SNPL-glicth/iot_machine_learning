"""Data models for Alpaca Account."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Position:
    """Posición abierta."""
    symbol: str
    side: str          # "long" o "short"
    qty: float         # Cantidad (positiva)
    avg_entry_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float
    lastday_price: float
    change_today: float
    asset_id: str
    asset_class: str
    exchange: str
    last_update: float = field(default_factory=time.time)


@dataclass
class AccountSnapshot:
    """Snapshot completo de la cuenta."""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    positions: Dict[str, Position]
    timestamp: float
