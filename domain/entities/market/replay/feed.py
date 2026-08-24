"""Feed histórico del Market Replay (FASE 5).

``HistoricalFeed`` es el contrato de entrada del replay: un iterador de
``MarketObservation`` **ordenado por timestamp no-decreciente** sobre un
único símbolo. La implementación concreta (CSV congelado, synthetic,
futuro WebSocket) es infraestructura; el engine solo consume el contrato.

El orden no-monótono se rechaza en el engine al recorrer el feed: un
feed que devuelve el pasado después del presente es un error, nunca un
estado a tolerar.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from ..observations import MarketObservation


class HistoricalFeed(Protocol):
    """Itera observaciones históricas ordenadas, sin mirar el futuro."""

    symbol: str
    resolution_seconds: int

    def iter_events(self) -> Iterator[MarketObservation]:
        """Itera las observaciones en orden no-decreciente de timestamp."""
        ...
