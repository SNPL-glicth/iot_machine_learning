"""Entidades de dominio ZENIN Market.

Observaciones y perfiles de proveedores, inmutables y sin dependencias
de infraestructura (regla: domain no conoce pymysql/sqlalchemy/redis/
providers/weaviate).
"""

from .capability import Capability, ProviderProfile
from .connection import ConnectionState
from .data_status import DataStatus
from .observations import (
    Candle,
    MarketObservation,
    OrderBookSnapshot,
    Quote,
    Trade,
)

__all__ = [
    "Capability",
    "ProviderProfile",
    "DataStatus",
    "MarketObservation",
    "Trade",
    "Quote",
    "Candle",
    "OrderBookSnapshot",
]
