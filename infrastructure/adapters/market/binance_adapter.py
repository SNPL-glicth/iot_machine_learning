"""BinanceAdapter (FASE 4) — payloads JSON de Binance -> entidades ZENIN.

Solo conversión (regla 10 de ARCHITECTURE.md).

Tipos de evento soportados:
    * ``aggTrade`` (WebSocket)                 -> ``Trade``
    * ``kline`` (WebSocket)                    -> ``Candle``
    * ``bookTicker`` (WebSocket, sin timestamp)-> ``Quote`` (usa ``received_at``)
    * snapshot REST ``GET /depth``             -> ``OrderBookSnapshot``

Timestamps: Binance entrega epoch en milisegundos (int); se convierten a
epoch float en segundos. Los eventos sin timestamp (bookTicker, depth)
usan ``received_at``: el momento de recepción declarado por el consumer;
si el consumer no lo provee, la conversión es imposible y se lanza
``ValueError`` (no se inventa el tiempo).
"""

from __future__ import annotations

from typing import Any, Mapping

from iot_machine_learning.domain.entities.market import MarketObservation
from .binance.trade import trade_from_payload
from .binance.quote import quote_from_payload
from .binance.candle import candle_from_payload
from .binance.order_book import order_book_from_payload
from .binance.constants import BINANCE_PROFILE


class BinanceAdapter:
    """Convierte eventos de mercado de Binance a entidades ZENIN."""

    provider_name = "binance"
    profile = BINANCE_PROFILE

    def parse(
        self,
        payload: Mapping[str, object],
        *,
        interval_seconds: int | None = None,
        symbol: str | None = None,
        received_at: float | None = None,
    ) -> MarketObservation:
        event_type = payload.get("e")
        if event_type == "aggTrade":
            return self.trade_from_payload(payload, received_at=received_at)
        if event_type == "kline":
            return self.candle_from_payload(
                payload,
                interval_seconds=interval_seconds,
                received_at=received_at,
            )
        if "lastUpdateId" in payload:
            return self.order_book_from_payload(
                payload, symbol=symbol, received_at=received_at
            )
        if event_type is None and "s" in payload and "b" in payload:
            return self.quote_from_payload(payload, received_at=received_at)
        raise ValueError(
            f"payload no reconocido de Binance: "
            f"e={event_type!r}, campos={sorted(payload)}"
        )

    # Public methods for direct access (backward compat with tests)
    def trade_from_payload(
        self,
        payload: Mapping[str, object],
        *,
        received_at: float | None = None,
    ):
        return trade_from_payload(
            payload,
            received_at=received_at,
            provider_name=self.provider_name,
            profile_has_realtime=self.profile.has_realtime,
        )

    def quote_from_payload(
        self,
        payload: Mapping[str, object],
        *,
        received_at: float | None = None,
    ):
        return quote_from_payload(
            payload,
            received_at=received_at,
            provider_name=self.provider_name,
            profile_has_realtime=self.profile.has_realtime,
        )

    def candle_from_payload(
        self,
        payload: Mapping[str, object],
        *,
        interval_seconds: int | None = None,
        received_at: float | None = None,
    ):
        return candle_from_payload(
            payload,
            interval_seconds=interval_seconds,
            received_at=received_at,
            provider_name=self.provider_name,
            profile_has_realtime=self.profile.has_realtime,
        )

    def order_book_from_payload(
        self,
        payload: Mapping[str, object],
        *,
        symbol: str | None = None,
        received_at: float | None = None,
    ):
        return order_book_from_payload(
            payload,
            symbol=symbol,
            received_at=received_at,
            provider_name=self.provider_name,
            profile_has_realtime=self.profile.has_realtime,
        )