"""Perfil estadístico auto-detectado de una serie temporal — Nivel 1.

UTSAE inspecciona la serie y construye un fingerprint numérico:
- ¿Es estacionaria o tiene drift?
- ¿Cuánta varianza tiene?
- ¿Hay periodicidad?
- ¿Cuántos puntos hay?
- ¿Cuál es el sampling rate?

Este perfil se usa para:
1. Elegir motor de predicción (en vez de elegir por sensor_id).
2. Ajustar hiperparámetros automáticamente.
3. Clasificar el "tipo" de serie sin saber el dominio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List


class VolatilityLevel(Enum):
    """Nivel de volatilidad de la serie."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StationarityHint(Enum):
    """Indicador de estacionariedad."""
    STATIONARY = "stationary"
    TREND = "trend"
    DRIFT = "drift"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SeriesProfile:
    """Fingerprint estadístico de una serie temporal.

    Attributes:
        n_points: Número de puntos en la serie.
        mean: Media aritmética.
        std: Desviación estándar.
        cv: Coeficiente de variación (std/|mean|).
        min_val: Valor mínimo observado.
        max_val: Valor máximo observado.
        range_val: Rango (max - min).
        mean_dt: Intervalo medio entre puntos.
        volatility: Nivel de volatilidad clasificado.
        stationarity: Indicador de estacionariedad.
        has_sufficient_data: True si hay datos suficientes para análisis avanzado.
    """

    n_points: int
    mean: float
    std: float
    cv: float
    min_val: float
    max_val: float
    range_val: float
    mean_dt: float
    volatility: VolatilityLevel
    stationarity: StationarityHint
    has_sufficient_data: bool


def compute_profile(values: List[float], timestamps: List[float]) -> SeriesProfile:
    """Computa el perfil estadístico de una serie temporal.

    Operación pura — sin I/O, sin estado, sin dominio.

    Args:
        values: Valores numéricos ordenados cronológicamente.
        timestamps: Timestamps correspondientes.

    Returns:
        SeriesProfile con el fingerprint de la serie.
    """
    n = len(values)
    if n == 0:
        return SeriesProfile(
            n_points=0, mean=0.0, std=0.0, cv=0.0,
            min_val=0.0, max_val=0.0, range_val=0.0, mean_dt=1.0,
            volatility=VolatilityLevel.LOW,
            stationarity=StationarityHint.UNKNOWN,
            has_sufficient_data=False,
        )

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    std = math.sqrt(variance)
    cv = std / abs(mean) if abs(mean) > 1e-10 else 0.0

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    # Mean dt
    if n >= 2 and len(timestamps) >= 2:
        mean_dt = (timestamps[-1] - timestamps[0]) / (n - 1)
    else:
        mean_dt = 1.0

    # Volatility classification
    volatility = _classify_volatility(cv, std, range_val)

    # Stationarity hint (simple slope test)
    stationarity = _detect_stationarity(values, std)

    has_sufficient = n >= 5

    return SeriesProfile(
        n_points=n, mean=mean, std=std, cv=cv,
        min_val=min_val, max_val=max_val, range_val=range_val,
        mean_dt=mean_dt, volatility=volatility,
        stationarity=stationarity,
        has_sufficient_data=has_sufficient,
    )


def _classify_volatility(cv: float, std: float, range_val: float) -> VolatilityLevel:
    """Clasifica volatilidad por coeficiente de variación."""
    if cv > 0.5 or (range_val > 0 and std / range_val > 0.3):
        return VolatilityLevel.HIGH
    if cv > 0.1:
        return VolatilityLevel.MEDIUM
    return VolatilityLevel.LOW


def _detect_stationarity(values: List[float], std: float) -> StationarityHint:
    """Detecta estacionariedad por pendiente normalizada (heurística rápida)."""
    n = len(values)
    if n < 5:
        return StationarityHint.UNKNOWN

    # Linear slope via least squares (simplified)
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))

    if den < 1e-12:
        return StationarityHint.STATIONARY

    slope = num / den

    # Normalize slope by std
    if std < 1e-10:
        return StationarityHint.STATIONARY

    normalized_slope = abs(slope) / std

    if normalized_slope > 0.3:
        return StationarityHint.TREND
    if normalized_slope > 0.1:
        return StationarityHint.DRIFT
    return StationarityHint.STATIONARY
