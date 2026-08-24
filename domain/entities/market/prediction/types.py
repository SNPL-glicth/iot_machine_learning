"""Tipos de apoyo de la predicción (FASE 3).

``Regime``, ``PredictionInterval`` e ``InputContext`` son valores
inmutables que acompañan a ``Prediction``; viven aparte para que la
entidad principal conserve una sola responsabilidad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Regime(Enum):
    """Régimen de mercado con el que se generó la predicción."""

    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    HIGH_VOLATILITY = "high_volatility"


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionInterval:
    """Intervalo de retorno esperado (fracciones, ej: -0.02 .. 0.03)."""

    lower: float
    upper: float
    confidence_level: float = 0.90

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.lower)
            or not math.isfinite(self.upper)
            or self.lower >= self.upper
        ):
            raise ValueError(
                f"intervalo inválido: lower={self.lower!r} upper={self.upper!r}"
            )
        if not math.isfinite(self.confidence_level) or not (
            0.0 < self.confidence_level <= 1.0
        ):
            raise ValueError(
                f"confidence_level inválido: {self.confidence_level!r}"
            )

    def contains(self, value: float) -> bool:
        """``True`` si el valor cae dentro de [lower, upper]."""
        return self.lower <= value <= self.upper


@dataclass(frozen=True, slots=True, kw_only=True)
class InputContext:
    """Contexto de entrada con el que se generó la predicción.

    Attributes:
        data_status: Estado de frescura de la observación de entrada.
        feature_count: Número de features que alimentaron el modelo.
        feature_version: Versión del feature-set (ej: "v3", "2026-08").
    """

    data_status: object | None = None  # DataStatus (import perezoso para evitar ciclo)
    feature_count: int | None = None
    feature_version: str | None = None

    def __post_init__(self) -> None:
        from ...market import DataStatus

        if self.data_status is not None and not isinstance(
            self.data_status, DataStatus
        ):
            raise TypeError("data_status debe ser DataStatus")
        if self.feature_count is not None and self.feature_count < 0:
            raise ValueError("feature_count debe ser >= 0")
        if self.feature_version is not None:
            version = self.feature_version.strip()
            if not version:
                raise ValueError("feature_version no puede ser vacío")
            object.__setattr__(self, "feature_version", version)
