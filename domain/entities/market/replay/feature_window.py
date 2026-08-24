"""Ventana de features del Market Replay (FASE 5).

Regla dura: la ventana solo contiene velas **cerradas** bajo el reloj
del replay (``candle.ts_close <= clock_now``). Wotany vela en formación
se retiene fuera hasta que el reloj la cierra; así la ventana jamás ve
el futuro, ni siquiera el minuto que está corriendo.

Las features derivadas son funciones puras de la ventana (deterministas:
mismas velas cerradas -> mismas features), lo que hace el test de oro
anti-look-ahead binario: cortar el feed después de la predicción no
puede cambiar la predicción.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..observations import Candle


@dataclass(frozen=True, slots=True)
class FeatureWindow:
    """Ventana de velas cerradas + derivados puros.

    Attributes:
        symbol: Símbolo de la ventana.
        candles: Velas cerradas ordenadas por timestamp (inmutable).
    """

    symbol: str
    candles: tuple[Candle, ...] = field(default_factory=tuple)

    def append_closed(self, candle: Candle) -> FeatureWindow:
        """Agrega una vela ya cerrada (la filtración la hace el engine)."""
        if not isinstance(candle, Candle):
            raise TypeError(f"esperaba Candle, obtenido {type(candle).__name__}")
        if self.candles and candle.timestamp <= self.candles[-1].timestamp:
            raise ValueError(
                "vela fuera de orden: la ventana solo acepta velas más nuevas"
            )
        return FeatureWindow(symbol=self.symbol, candles=self.candles + (candle,))

    @property
    def size(self) -> int:
        return len(self.candles)

    def last_closed(self, at_or_before: float | None = None) -> Candle | None:
        """Última vela cerrada a lo sumo en ``at_or_before``."""
        if not self.candles:
            return None
        if at_or_before is None:
            return self.candles[-1]
        for candle in reversed(self.candles):
            if candle.timestamp <= at_or_before:
                return candle
        return None

    def last_close(self, at_or_before: float | None = None) -> float | None:
        candle = self.last_closed(at_or_before)
        return candle.close if candle is not None else None

    def window(self, size: int) -> FeatureWindow:
        """Sub-ventana con las últimas ``size`` velas."""
        return FeatureWindow(symbol=self.symbol, candles=self.candles[-size:])

    def returns(self, size: int) -> tuple[float, ...]:
        """Retornos logarítmicos de las últimas ``size`` velas cerradas."""
        if len(self.candles) < size + 1:
            raise ValueError(
                f"ventana insuficiente: se requieren {size + 1} velas, hay {self.size}"
            )
        candles = self.candles[-(size + 1) :]
        out: list[float] = []
        for prev, curr in zip(candles, candles[1:], strict=False):
            if prev.close <= 0 or curr.close <= 0:
                raise ValueError("closes deben ser > 0 para retornos logarítmicos")
            out.append(math.log(curr.close / prev.close))
        return tuple(out)

    def mean_return(self, size: int) -> float:
        """Media de los retornos de la ventana (drift corto - determinista)."""
        r = self.returns(size)
        return sum(r) / len(r)

    def std_return(self, size: int) -> float:
        """Desviación estándar poblacional de retornos de la ventana."""
        r = self.returns(size)
        mean = sum(r) / len(r)
        variance = sum((x - mean) ** 2 for x in r) / len(r)
        return math.sqrt(variance)

    def vwap(self, size: int) -> float:
        """VWAP (por volumen) de las últimas ``size`` velas cerradas."""
        candles = self.candles[-size:]
        total_vol: float = sum(c.volume for c in candles)
        if total_vol <= 0:
            raise ValueError("volumen total debe ser > 0 para vwap")
        price_volume: float = sum(
            (c.high + c.low + c.close) / 3.0 * c.volume for c in candles
        )
        return price_volume / total_vol

    def typical_range(self, size: int) -> float:
        """Rango típico medio (|high - low| promedio) de la ventana."""
        candles = self.candles[-size:]
        if not candles:
            raise ValueError("ventana vacía")
        total_range: float = sum(c.high - c.low for c in candles)
        return total_range / len(candles)
