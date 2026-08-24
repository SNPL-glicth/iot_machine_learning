"""Market Observations — Base class."""

from __future__ import annotations

from dataclasses import dataclass

from ..data_status import DataStatus
from ..validators import validate_symbol, validate_timestamp


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketObservation:
    """Base inmutable de toda observación de mercado.

    Attributes:
        symbol: Símbolo/instrumento (ej: "NVDA", "BTC/USD").
        timestamp: Epoch en segundos con precisión sub-segundo.
        data_status: Estado de frescura/origen de la observación.
        source_provider: Proveedor de origen (ej: "alpaca", "binance").
        venue: Venue/exchange si el proveedor lo reporta (``None`` si no).
    """

    symbol: str
    timestamp: float
    data_status: DataStatus
    source_provider: str
    venue: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", validate_symbol(self.symbol))
        validate_timestamp(self.timestamp)
        if not isinstance(self.data_status, DataStatus):
            raise TypeError(
                f"data_status debe ser DataStatus, no {self.data_status!r}"
            )
        provider = self.source_provider.strip()
        if not provider:
            raise ValueError("source_provider no puede ser vacío")
        object.__setattr__(self, "source_provider", provider)
        if self.venue is not None:
            venue = self.venue.strip()
            if not venue:
                raise ValueError("venue no puede ser vacío")
            object.__setattr__(self, "venue", venue)

    @property
    def is_live(self) -> bool:
        """``True`` si la observación es una señal live."""
        return self.data_status.is_live_signal