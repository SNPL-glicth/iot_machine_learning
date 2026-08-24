"""Market Observations — Quote."""

from __future__ import annotations

from dataclasses import dataclass, field

from .market_observation import MarketObservation
from ..validators import validate_price, validate_size


@dataclass(frozen=True, slots=True, kw_only=True)
class Quote(MarketObservation):
    """Mejor bid/ask (top-of-book) para un instrumento.

    Nota: no se impone bid <= ask porque libros cruzados existen
    temporalmente en feeds reales; la validación rechaza estados
    imposibles (no finitos, <= 0), no estados que los venues reportan.
    """

    bid: float
    bid_size: float
    ask: float
    ask_size: float
    bid_exchange: str | None = None
    ask_exchange: str | None = None
    conditions: tuple[str, ...] = field(default_factory=tuple)
    tape: str | None = None

    def __post_init__(self) -> None:
        MarketObservation.__post_init__(self)
        validate_price(self.bid, "bid")
        validate_price(self.ask, "ask")
        validate_size(self.bid_size, "bid_size")
        validate_size(self.ask_size, "ask_size")

    @property
    def spread(self) -> float:
        """Diferencia ask - bid (puede ser negativa en libros cruzados)."""
        return self.ask - self.bid

    @property
    def midpoint(self) -> float:
        """Punto medio bid/ask."""
        return (self.bid + self.ask) / 2.0