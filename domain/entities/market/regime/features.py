"""Features de régimen sobre retornos cerrados (FASE 2).

Funciones puras: mismos retornos → mismas features. Todo se calcula
sobre velas ya cerradas (la ventana jamás ve el futuro).

Vector (5 dims): (abs_trend_z, vol_ln, autocorr_lag1, breakout_z, downside)
- abs_trend_z: |drift|/vol·√n, saturado a 3.0 (agnóstico a dirección).
- vol_ln: ln(vol/0.006); 0.006 ≈ media geométrica de los umbrales
  deterministas 0.004/0.010 de ``replay/regime.py`` (puente documentado).
- autocorr_lag1: [-1, 1]; negativa ⇒ mean-reversion, positiva ⇒ trend.
- breakout_z: max|retorno demeaned|/vol (expansión de rango).
- downside: |neg|/(|neg|+|pos|) en [0,1]; 0.5 simétrico, →1 sangría.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["RegimeFeatures", "regime_features", "MIN_RETURNS"]

MIN_RETURNS: int = 10
VOL_REFERENCE: float = 0.006
TREND_SATURATION: float = 3.0


@dataclass(frozen=True, slots=True, kw_only=True)
class RegimeFeatures:
    """Vector de features de una ventana de retornos."""

    abs_trend: float
    vol_ln: float
    autocorr: float
    breakout: float
    downside: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.abs_trend, self.vol_ln, self.autocorr,
                self.breakout, self.downside)


def regime_features(returns: tuple[float, ...] | list[float]) -> RegimeFeatures:
    """Extrae el vector de régimen de una serie de retornos."""
    returns = tuple(returns)
    n = len(returns)
    if n < MIN_RETURNS:
        raise ValueError(
            f"retornos insuficientes para régimen: {n} < {MIN_RETURNS}"
        )
    if any(not math.isfinite(r) for r in returns):
        raise ValueError("retornos no finitos")

    drift = sum(returns) / n
    var = sum((r - drift) ** 2 for r in returns) / n
    vol = math.sqrt(var) if var > 0 else 1e-12

    trend = abs(drift) / vol * math.sqrt(n)
    vol_ln = math.log(vol / VOL_REFERENCE)

    mean = drift
    num = sum((returns[i] - mean) * (returns[i + 1] - mean) for i in range(n - 1))
    den = sum((r - mean) ** 2 for r in returns)
    autocorr = (num / den) if den > 0 else 0.0
    autocorr = max(-1.0, min(1.0, autocorr))

    breakout = max(abs(r - drift) for r in returns) / vol

    neg = sum(abs(r) for r in returns if r < 0)
    pos = sum(r for r in returns if r > 0)
    downside = (neg / (neg + pos)) if (neg + pos) > 0 else 0.5

    return RegimeFeatures(
        abs_trend=min(trend, TREND_SATURATION),
        vol_ln=vol_ln,
        autocorr=autocorr,
        breakout=breakout,
        downside=downside,
    )
