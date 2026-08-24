"""Market Observations — Candle."""

from __future__ import annotations

from dataclasses import dataclass, field

from .market_observation import MarketObservation
from ..validators import validate_price, validate_size


@dataclass(frozen=True, slots=True, kw_only=True)
class Candle(MarketObservation):
    """Vela OHLCV (propia del dominio o del provider).

    Attributes:
        open/high/low/close: Precios OHLC (close puede ser <= 0 en casos
            degenerados? No: siempre > 0 en mercados reales).
        volume: Volumen de la vela (>= 0).
        vwap: Precio promedio ponderado por volumen si está disponible.
        trade_count: Número de operaciones que compusieron la vela.
        interval_seconds: Duración de la vela en segundos.
        adjusted: ``True`` si los precios están ajustados por splits/dividendos.
    """

    open: float
    high: float
    low: float
    close: float
    volume: float
    interval_seconds: int
    vwap: float | None = None
    trade_count: int | None = None
    adjusted: bool = False

    def __post_init__(self) -> None:
        MarketObservation.__post_init__(self)
        validate_price(self.open, "open")
        validate_price(self.high, "high")
        validate_price(self.low, "low")
        validate_price(self.close, "close")
        validate_size(self.volume, "volume")
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds debe ser > 0")
        if self.low > min(self.open, self.close):
            raise ValueError(f"low > min(open, close): low={self.low}")
        if self.high < max(self.open, self.close):
            raise ValueError(f"high < max(open, close): high={self.high}")
        if self.vwap is not None:
            validate_price(self.vwap, "vwap")
        if self.trade_count is not None and self.trade_count < 0:
            raise ValueError("trade_count debe ser >= 0")

    @property
    def body(self) -> float:
        """Diferencia close - open."""
        return self.close - self.open

    @property
    def is_bullish(self) -> bool:
        """``True`` si la vela cerró por encima del open."""
        return self.close > self.open