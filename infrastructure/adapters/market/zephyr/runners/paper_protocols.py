"""Protocols and low-level feed adaptors for PaperBotRunner."""

from __future__ import annotations

from typing import Any, Protocol

from iot_machine_learning.domain.entities.market import Candle
from iot_machine_learning.domain.entities.market.prediction.prediction import (
    Prediction,
)


class PredictionRepoProtocol(Protocol):
    """Subconjunto del contrato de MarketPredictionRepository que usa el runner."""

    def save_batch(self, predictions) -> int: ...

    def pending_outcomes(self, *, symbol: str | None = None): ...


class EvidenceRepoProtocol(Protocol):
    """Subconjunto del contrato de CalibrationEvidenceRepository."""

    def save_batch(self, records) -> int: ...


class FeedProtocol(Protocol):
    """Subconjunto del contrato de BinanceKlinesFeed."""

    def poll_closed(self) -> tuple[Candle, ...]: ...

    def recent_candles(self, limit: int | None = None) -> tuple[Candle, ...]: ...

    def last_close(self, at_or_before: float) -> float | None: ...

    @property
    def connected(self) -> bool: ...


class _StaticCandleFeed:
    """Feed mínimo sobre una tupla de velas (contrato HistoricalFeed)."""

    def __init__(self, candles: tuple[Candle, ...], symbol: str = "", resolution_seconds: int = 0) -> None:
        self._candles = candles
        self.symbol = symbol
        self.resolution_seconds = resolution_seconds

    def iter_events(self):
        yield from self._candles


def row_to_prediction_safe(row: Any) -> Prediction:
    """Wrapper del mapper del repo para mantener este módulo legible."""
    from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_prediction_repository import (
        row_to_prediction,
    )

    return row_to_prediction(row)
