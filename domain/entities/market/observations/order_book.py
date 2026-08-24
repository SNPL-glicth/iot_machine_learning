"""Market Observations — OrderBookSnapshot."""

from __future__ import annotations

from dataclasses import dataclass

from .market_observation import MarketObservation
from ..validators import validate_price, validate_size


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderBookSnapshot(MarketObservation):
    """Snapshot del libro de órdenes (solo si el provider tiene L2).

    Attributes:
        bids: Niveles (price, size) ordenados descendente por precio.
        asks: Niveles (price, size) ordenados ascendente por precio.
        reset: ``True`` si el snapshot reemplaza el libro local completo.
    """

    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    reset: bool = False

    def __post_init__(self) -> None:
        MarketObservation.__post_init__(self)
        bids = tuple(self.bids)
        asks = tuple(self.asks)
        if not bids and not asks:
            raise ValueError("OrderBookSnapshot no puede tener libro vacío")
        for level in bids:
            validate_price(level[0], "bid price")
            validate_size(level[1], "bid size")
        for level in asks:
            validate_price(level[0], "ask price")
            validate_size(level[1], "ask size")
        self._validate_ordering(bids, descending=True, name="bids")
        self._validate_ordering(asks, descending=False, name="asks")
        object.__setattr__(self, "bids", bids)
        object.__setattr__(self, "asks", asks)

    @staticmethod
    def _validate_ordering(
        levels: tuple[tuple[float, float], ...],
        *,
        descending: bool,
        name: str,
    ) -> None:
        for prev, curr in zip(levels, levels[1:], strict=False):
            if descending and curr[0] >= prev[0]:
                raise ValueError(
                    f"{name} debe estar ordenado descendente por precio: "
                    f"{prev[0]} -> {curr[0]}"
                )
            if not descending and curr[0] <= prev[0]:
                raise ValueError(
                    f"{name} debe estar ordenado ascendente por precio: "
                    f"{prev[0]} -> {curr[0]}"
                )

    @property
    def best_bid(self) -> float | None:
        """Mejor precio bid."""
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Mejor precio ask."""
        return self.asks[0][0] if self.asks else None

    @property
    def imbalance(self) -> float:
        """Desbalance de volumen bid vs ask en (-1, 1); 0 si no hay datos."""
        bid_vol = sum(level[1] for level in self.bids)
        ask_vol = sum(level[1] for level in self.asks)
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total