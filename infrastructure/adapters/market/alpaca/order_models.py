"""Data models and response parser for Alpaca orders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class OrderRequest:
    """Solicitud de orden estandarizada."""
    symbol: str
    side: str              # "buy" o "sell"
    order_type: str        # "market", "limit", "stop", "stop_limit", "trailing_stop"
    qty: float = 0.0
    notional: Optional[float] = None
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_price: Optional[float] = None
    trail_percent: Optional[float] = None
    time_in_force: str = "day"  # day, gtc, opg, cls, ioc, fok
    extended_hours: bool = False
    client_order_id: Optional[str] = None
    order_class: Optional[str] = None  # simple, bracket, oco, oto
    take_profit: Optional[Dict] = None
    stop_loss: Optional[Dict] = None


@dataclass
class OrderResponse:
    """Respuesta de orden de Alpaca."""
    id: str
    client_order_id: str
    symbol: str
    status: str
    side: str
    order_type: str
    qty: float
    price: Optional[float]
    stop_price: Optional[float]
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    commission: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    raw: Dict = field(default_factory=dict)


def parse_order_response(data: Dict[str, Any]) -> OrderResponse:
    """Parsea el diccionario retornado por Alpaca a OrderResponse."""
    return OrderResponse(
        id=data.get("id", ""),
        client_order_id=data.get("client_order_id", ""),
        symbol=data.get("symbol", ""),
        status=data.get("status", ""),
        side=data.get("side", ""),
        order_type=data.get("order_type", ""),
        qty=float(data.get("qty") or 0),
        price=float(data.get("limit_price", 0)) if data.get("limit_price") else None,
        stop_price=float(data.get("stop_price", 0)) if data.get("stop_price") else None,
        filled_qty=float(data.get("filled_qty") or 0),
        filled_avg_price=float(data.get("filled_avg_price", 0)) if data.get("filled_avg_price") else None,
        commission=float(data.get("commission", 0)) if data.get("commission") else 0.0,
        created_at=data.get("created_at", 0),
        updated_at=data.get("updated_at", 0),
        raw=data,
    )
