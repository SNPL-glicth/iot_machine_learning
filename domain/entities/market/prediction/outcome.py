"""Outcome de una predicción (FASE 3).

Resultado realizado al vencimiento del horizonte. Se mide contra el
mismo símbolo y el mismo horizonte de su predicción; la asociación
equivocada (otro símbolo, otro horizonte) se rechaza al vincularlo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class Outcome:
    """Desenlace de una predicción al cerrarse su horizonte.

    Attributes:
        symbol: Símbolo del resultado (debe coincidir con la observación).
        observation_timestamp: Inicio del período (timestamp de la observación).
        horizon_seconds: Horizonte que cubre (debe coincidir con la predicción).
        measured_at: Momento en que se midió el resultado.
        final_price: Precio al cierre del período.
        return_realized: Retorno realizado como fracción
            ((final_price - ref_price) / ref_price).
    """

    symbol: str
    observation_timestamp: float
    horizon_seconds: int
    measured_at: float
    final_price: float
    return_realized: float

    def __post_init__(self) -> None:
        symbol = self.symbol.strip()
        if not symbol:
            raise ValueError("symbol no puede ser vacío")
        object.__setattr__(self, "symbol", symbol)

        if self.horizon_seconds <= 0:
            raise ValueError(f"horizon_seconds debe ser > 0: {self.horizon_seconds}")

        if not math.isfinite(self.measured_at) or self.measured_at <= 0:
            raise ValueError(f"measured_at inválido: {self.measured_at!r}")
        horizon_end = self.observation_timestamp + self.horizon_seconds
        if self.measured_at < horizon_end:
            raise ValueError(
                "measured_at anterior al vencimiento del horizonte: "
                f"{self.measured_at} < {horizon_end}"
            )

        if not math.isfinite(self.final_price) or self.final_price <= 0:
            raise ValueError(f"final_price inválido: {self.final_price!r}")
        if not math.isfinite(self.return_realized):
            raise ValueError(f"return_realized inválido: {self.return_realized!r}")

    @classmethod
    def from_prices(
        cls,
        *,
        symbol: str,
        ref_timestamp: float,
        ref_price: float,
        horizon_seconds: int,
        final_price: float,
        measured_at: float | None = None,
    ) -> Outcome:
        """Construye el Outcome desde el precio de referencia y el final.

        ``return_realized`` se calcula como ``(final - ref) / ref``.
        ``measured_at`` por defecto: exactamente al vencimiento del horizonte.
        """
        if not math.isfinite(ref_price) or ref_price <= 0:
            raise ValueError(f"ref_price inválido: {ref_price!r}")
        if measured_at is None:
            measured_at = ref_timestamp + horizon_seconds
        return cls(
            symbol=symbol,
            observation_timestamp=ref_timestamp,
            horizon_seconds=horizon_seconds,
            measured_at=measured_at,
            final_price=final_price,
            return_realized=(final_price - ref_price) / ref_price,
        )
