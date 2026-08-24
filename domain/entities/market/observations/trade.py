"""Market Observations — Trade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from .market_observation import MarketObservation
from ..data_status import DataStatus
from ..validators import validate_price, validate_size


@dataclass(frozen=True, slots=True, kw_only=True)
class Trade(MarketObservation):
    """Operación ejecutada (tick de transacción).

    Attributes:
        price: Precio de la operación.
        size: Cantidad operada.
        trade_id: ID de la operación si el proveedor lo reporta.
        taker_side: Lado agresor ("buy"/"sell") — cripto; ``None`` si no.
        conditions: Condiciones de tape si el proveedor las reporta.
        tape: Cinta consolidada ("A"/"B"/"C") si aplica.
        corrected: ``True`` si el proveedor marcó cancel/corrección.
    """

    price: float
    size: float
    trade_id: str | None = None
    taker_side: str | None = None
    conditions: tuple[str, ...] = field(default_factory=tuple)
    tape: str | None = None
    corrected: bool = False

    def __post_init__(self) -> None:
        MarketObservation.__post_init__(self)
        validate_price(self.price, "price")
        validate_size(self.size, "size")
        if self.size == 0:
            raise ValueError("size no puede ser 0 en un Trade")
        if self.conditions:
            for cond in self.conditions:
                if not isinstance(cond, str) or not cond:
                    raise TypeError("conditions debe ser tupla de str no vacías")
        if self.taker_side is not None and self.taker_side not in ("buy", "sell"):
            raise ValueError(f"taker_side inválido: {self.taker_side!r}")