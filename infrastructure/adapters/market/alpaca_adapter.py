"""AlpacaAdapter (FASE 4) — payloads JSON de Alpaca -> entidades ZENIN.

Solo conversión (regla 10 de ARCHITECTURE.md): el adapter traduce el
evento del proveedor a una entidad del dominio; nunca inventa datos
que el payload no trae.

Tipos de evento soportados (WebSocket stream, campo ``T``):
    * ``t`` -> ``Trade``
    * ``q`` -> ``Quote``
    * ``b`` -> ``Candle`` (el stream de bars no anuncia intervalo: el
      consumer lo aporta vía ``interval_seconds``)

Los feeds históricos REST reutilizan esta misma normalización: itere los
items del contenedor (``trades``/``quotes``/``bars``) y llame a la
función correspondiente; para bars históricos pase ``interval_seconds``.

Timestamps: Alpaca entrega ISO-8601 UTC (con ``Z`` o ``+00:00``) con
precisión de nanosegundos; se convierten a epoch float en segundos
(fracción conservada por el float).
"""

from __future__ import annotations

from typing import Any, Mapping

from iot_machine_learning.domain.entities.market import MarketObservation
from .alpaca.trade import trade_from_payload
from .alpaca.quote import quote_from_payload
from .alpaca.candle import candle_from_payload
from .alpaca.constants import ALPACA_PROFILE


class AlpacaAdapter:
    """Convierte eventos de mercado de Alpaca a entidades ZENIN."""

    provider_name = "alpaca"
    profile = ALPACA_PROFILE

    def parse(
        self,
        payload: Mapping[str, object],
        *,
        interval_seconds: int | None = None,
        symbol: str | None = None,
        received_at: float | None = None,
    ) -> MarketObservation:
        event_type = payload.get("T")
        if event_type == "t":
            return self.trade_from_payload(payload, received_at=received_at)
        if event_type == "q":
            return self.quote_from_payload(payload, received_at=received_at)
        if event_type == "b":
            return self.candle_from_payload(
                payload,
                interval_seconds=interval_seconds,
                received_at=received_at,
            )
        raise ValueError(
            f"payload no reconocido de Alpaca: campo 'T' esperado "
            f"('t'|'q'|'b'), obtenido {event_type!r}"
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
    ) -> object:
        raise NotImplementedError(
            "Alpaca no soporta ORDER_BOOK_L2 (Capability ausente del profile)"
        )