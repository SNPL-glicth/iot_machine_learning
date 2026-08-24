"""Port MarketDataProvider (FASE 4).

Contrato que toda fuente de datos de mercado implementa. Dependencias
hacia adentro: los adaptadores (infrastructure) implementan este port y
traducen payloads del proveedor a entidades ZENIN; el dominio nunca
conoce Alpaca/Binance.

Responsabilidad del adapter (regla 10 de ARCHITECTURE.md):

    provider payload -> normalización -> entidad ZENIN

Nunca al revés. El puerto es deliberadamente sobre conversión: el
streaming, la reconexión y la cola de eventos son capas posteriores.

Timestamps:
    * Payloads con timestamp propio (Alpaca ISO-8601, Binance epoch ms)
      se convierten a epoch float segundos.
    * Payloads SIN timestamp (ej. bookTicker de Binance) usan
      ``received_at`` (momento de recepción del consumer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..entities.market import (
        Candle,
        MarketObservation,
        OrderBookSnapshot,
        ProviderProfile,
        Quote,
        Trade,
    )


@runtime_checkable
class MarketDataProvider(Protocol):
    """Protocolo mínimo de conversión de eventos de mercado."""

    provider_name: str
    profile: ProviderProfile

    def parse(
        self,
        payload: Mapping[str, object],
        *,
        interval_seconds: int | None = None,
        symbol: str | None = None,
        received_at: float | None = None,
    ) -> MarketObservation:
        """Despacha un payload crudo al tipo de observación correcto.

        Lanza ValueError si el payload no pertenece a este proveedor.
        ``symbol`` se reenvía a los eventos que no lo declaran en el
        cuerpo (snapshots de depth REST).
        """
        ...

    def trade_from_payload(
        self,
        payload: Mapping[str, object],
        *,
        received_at: float | None = None,
    ) -> Trade:
        """Convierte un evento de operación en ``Trade``."""
        ...

    def quote_from_payload(
        self,
        payload: Mapping[str, object],
        *,
        received_at: float | None = None,
    ) -> Quote:
        """Convierte un evento top-of-book en ``Quote``."""
        ...

    def candle_from_payload(
        self,
        payload: Mapping[str, object],
        *,
        interval_seconds: int | None = None,
        received_at: float | None = None,
    ) -> Candle:
        """Convierte un evento de vela en ``Candle``.

        ``interval_seconds`` lo aporta el consumer cuando el payload no
        lo trae (ej. bars de Alpaca); si el payload lo declara (ej.
        klines de Binance) se deriva y el parámetro es opcional.
        """
        ...

    def order_book_from_payload(
        self,
        payload: Mapping[str, object],
        *,
        symbol: str | None = None,
        received_at: float | None = None,
    ) -> OrderBookSnapshot:
        """Convierte un snapshot de libro en ``OrderBookSnapshot``.

        Reservado a proveedores con ``ORDER_BOOK_L2`` (ej. Binance).

        ``symbol`` lo aporta el consumer cuando el payload no lo
        declara (el endpoint REST de depth de Binance recibe el símbolo
        por la URL, no en el cuerpo).
        """
        ...
