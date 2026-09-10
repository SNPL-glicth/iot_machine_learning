"""Distribución condicional de retornos (FASE 1).

No "SPY va a subir" sino: dadas estas condiciones, durante los próximos
N minutos el retorno esperado es +0.18% con esta distribución.

VO inmutable y puro: la forma paramétrica es gaussiana
(mean=expected_return, std=volatility) para ``p_exceed``/``p_drawdown``;
los cuantiles q10/q50/q90 son los que el modelo declara y lo que se
evalúa (pinball, CRPS, cobertura). Si el modelo no es gaussiano, los
cuantiles mandan y la gaussiana es solo aproximación de colas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

__all__ = ["DISTRIBUTIONAL_HORIZONS", "QUANTILE_LEVELS", "ReturnDistribution"]

#: Horizontes del paper loop para predicción distribucional (1m/5m/15m).
DISTRIBUTIONAL_HORIZONS: Final = (60, 300, 900)

#: Niveles de cuantil estándar que toda distribución declara.
QUANTILE_LEVELS: Final = (0.10, 0.50, 0.90)

_SQRT2: Final = math.sqrt(2.0)


def _phi(z: float) -> float:
    """CDF normal estándar (via erf, stdlib)."""
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


@dataclass(frozen=True, slots=True, kw_only=True)
class ReturnDistribution:
    """Distribución futura de retornos como fracciones (0.0018 = 0.18%).

    Attributes:
        expected_return: Media del retorno al horizonte.
        volatility: Desvío estándar (> 0).
        quantile_10/50/90: Cuantiles declarados (ordenados).
        expected_holding_seconds: Tenencia esperada (None si buy&hold
            hasta el horizonte).
        cvar_05: Cola esperada al 5% (fracción, típicamente <= 0).
    """

    expected_return: float
    volatility: float
    quantile_10: float
    quantile_50: float
    quantile_90: float
    expected_holding_seconds: int | None = None
    cvar_05: float | None = None

    def __post_init__(self) -> None:
        for name in ("expected_return", "volatility", "quantile_10",
                     "quantile_50", "quantile_90"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} inválido: {value!r}")
        if self.volatility <= 0:
            raise ValueError(
                f"volatility debe ser > 0: {self.volatility!r}"
            )
        if not (self.quantile_10 <= self.quantile_50 <= self.quantile_90):
            raise ValueError(
                "cuantiles desordenados: "
                f"q10={self.quantile_10!r} q50={self.quantile_50!r} "
                f"q90={self.quantile_90!r}"
            )
        if not (self.quantile_10 <= self.expected_return <= self.quantile_90):
            raise ValueError(
                "expected_return fuera de la banda q10..q90: "
                f"{self.expected_return!r} ∉ "
                f"[{self.quantile_10!r}, {self.quantile_90!r}]"
            )
        if self.expected_holding_seconds is not None:
            if not isinstance(self.expected_holding_seconds, int):
                raise TypeError("expected_holding_seconds debe ser int")
            if self.expected_holding_seconds <= 0:
                raise ValueError("expected_holding_seconds debe ser > 0")
        if self.cvar_05 is not None and not math.isfinite(self.cvar_05):
            raise ValueError(f"cvar_05 inválido: {self.cvar_05!r}")

    def p_exceed(self, threshold: float) -> float:
        """P(retorno > threshold) bajo aproximación gaussiana."""
        if not math.isfinite(threshold):
            raise ValueError(f"threshold inválido: {threshold!r}")
        z = (threshold - self.expected_return) / self.volatility
        return 1.0 - _phi(z)

    def p_drawdown(self, threshold: float) -> float:
        """P(retorno <= -threshold); threshold > 0 (ej: 0.01 = -1%)."""
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"threshold debe ser > 0: {threshold!r}")
        return 1.0 - self.p_exceed(-threshold)

    @property
    def implied_probability_up(self) -> float:
        """P(retorno > 0) implicada por la distribución."""
        return self.p_exceed(0.0)

    def within_band(self, value: float) -> bool:
        """``True`` si el realizado cayó en [q10, q90]."""
        return self.quantile_10 <= value <= self.quantile_90
