"""Enums y perfiles de proveedores de datos de mercado.

Contract v1 — ``Capability`` describe qué puede entregar un provider;
``ProviderProfile`` expresa un proveedor concreto con sus capacidades.
El dominio usa las capacidades para no asumir información inexistente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Capability(Enum):
    """Capacidades que un proveedor de datos puede ofrecer."""

    TRADES = "trades"
    QUOTES = "quotes"
    CANDLES = "candles"
    ORDER_BOOK_L2 = "order_book_l2"
    HISTORICAL_TICKS = "historical_ticks"
    HISTORICAL_BARS = "historical_bars"
    REALTIME = "realtime"
    DELAYED = "delayed"
    VWAP = "vwap"
    TRADE_CONDITIONS = "trade_conditions"
    NANOSECOND_TIMESTAMP = "nanosecond_timestamp"
    SECOND_AGGREGATES = "second_aggregates"
    ADJUSTED_SERIES = "adjusted_series"


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """Perfil inmutable de un proveedor de datos de mercado.

    Attributes:
        provider: Identificador del proveedor (ej: "alpaca", "binance").
        asset_class: Clase de activo (ej: "equities", "crypto").
        capabilities: Conjunto de capacidades soportadas.
        max_ws_symbols: Máximo de símbolos simultáneos por WebSocket
            (``None`` si ilimitado o desconocido).
    """

    provider: str
    asset_class: str
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    max_ws_symbols: int | None = None

    def __post_init__(self) -> None:
        provider = self.provider.strip()
        asset_class = self.asset_class.strip()
        if not provider:
            raise ValueError("provider no puede ser vacío")
        if not asset_class:
            raise ValueError("asset_class no puede ser vacío")
        if self.max_ws_symbols is not None and self.max_ws_symbols < 1:
            raise ValueError("max_ws_symbols debe ser >= 1 o None")

        capabilities = frozenset(self.capabilities)
        for cap in capabilities:
            if not isinstance(cap, Capability):
                raise TypeError(
                    f"capacidad inválida: {cap!r} (debe ser Capability)"
                )

        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "asset_class", asset_class)
        object.__setattr__(self, "capabilities", capabilities)

    @property
    def has_order_book(self) -> bool:
        """``True`` si el proveedor entrega profundidad de libro (L2)."""
        return Capability.ORDER_BOOK_L2 in self.capabilities

    @property
    def has_realtime(self) -> bool:
        """``True`` si el proveedor entrega feed en vivo."""
        return Capability.REALTIME in self.capabilities
