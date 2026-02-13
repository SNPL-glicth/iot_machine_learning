"""Cómputo de análisis estructural unificado para series de tiempo.

Responsabilidad ÚNICA: calcular slope, curvature, stability, noise_ratio,
régimen y métricas agregadas a partir de valores + timestamps.

Función pura — sin I/O, sin estado, sin dependencias de infraestructura.
Usado por TimeSeries.structural_analysis y SensorWindow.structural_analysis.

Unifica cálculos que antes estaban dispersos en:
- taylor/diagnostics.py (accel_variance, stability_indicator)
- taylor/derivatives.py (slope, curvature via backward differences)
- series_profile.py (stationarity, volatility)
- cognitive/orchestrator.py (_classify_regime)
"""

from __future__ import annotations

import math
from typing import List

from ..entities.structural_analysis import (
    RegimeType,
    StructuralAnalysis,
    _classify_regime,
)


def compute_structural_analysis(
    values: List[float],
    timestamps: List[float],
) -> StructuralAnalysis:
    """Computa análisis estructural unificado de una serie.

    Calcula en un único pase: slope (f'), curvature (f''), stability,
    accel_variance, noise_ratio, régimen, y estadísticas básicas.

    Args:
        values: Valores numéricos ordenados cronológicamente.
        timestamps: Timestamps correspondientes (misma longitud).

    Returns:
        ``StructuralAnalysis`` con todas las métricas estructurales.

    Raises:
        ValueError: Si values y timestamps tienen diferente longitud.
    """
    n = len(values)

    if len(timestamps) != n:
        raise ValueError(
            f"values ({n}) y timestamps ({len(timestamps)}) "
            f"deben tener la misma longitud"
        )

    if n < 2:
        return StructuralAnalysis(n_points=n, mean=values[0] if n == 1 else 0.0)

    # --- Basic statistics ---
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    std = math.sqrt(variance)
    noise_ratio = std / abs(mean) if abs(mean) > 1e-9 else 0.0

    # --- Δt ---
    dt = _compute_median_dt(timestamps)

    # --- Slope (f') via backward difference at last point ---
    slope = (values[-1] - values[-2]) / dt

    # --- Curvature (f'') via backward difference ---
    curvature = 0.0
    if n >= 3:
        curvature = (values[-1] - 2.0 * values[-2] + values[-3]) / (dt * dt)

    # --- Acceleration variance (across the window) ---
    accel_variance = _compute_accel_variance(values, dt)

    # --- Stability indicator ---
    stability = _compute_stability(accel_variance, values[-1])

    # --- Trend strength ---
    mean_ref = abs(mean) if abs(mean) > 1e-9 else 1.0
    trend_strength = abs(slope) / mean_ref

    # --- Regime classification ---
    regime = _classify_regime(noise_ratio, slope, std, mean)

    return StructuralAnalysis(
        slope=slope,
        curvature=curvature,
        stability=stability,
        accel_variance=accel_variance,
        noise_ratio=noise_ratio,
        regime=regime,
        mean=mean,
        std=std,
        trend_strength=trend_strength,
        n_points=n,
        dt=dt,
    )


def _compute_median_dt(timestamps: List[float]) -> float:
    """Calcula Δt mediano (robusto a outliers en timestamps).

    Usa la mediana en vez de la media para ser robusto a gaps.
    """
    if len(timestamps) < 2:
        return 1.0

    dts = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    dts = [dt for dt in dts if dt > 1e-9]

    if not dts:
        return 1.0

    sorted_dts = sorted(dts)
    mid = len(sorted_dts) // 2
    if len(sorted_dts) % 2 == 0:
        return (sorted_dts[mid - 1] + sorted_dts[mid]) / 2.0
    return sorted_dts[mid]


def _compute_accel_variance(values: List[float], dt: float) -> float:
    """Varianza poblacional de la segunda derivada a lo largo de la ventana.

    Mide cuánto cambia la aceleración — proxy de cuán bien un
    polinomio de bajo orden aproxima la señal.
    """
    n = len(values)
    if n < 4:
        return 0.0

    dt_sq = dt * dt
    accels: List[float] = []
    for i in range(2, n):
        accel = (values[i] - 2.0 * values[i - 1] + values[i - 2]) / dt_sq
        accels.append(accel)

    if len(accels) < 2:
        return 0.0

    mean_a = sum(accels) / len(accels)
    return sum((a - mean_a) ** 2 for a in accels) / len(accels)


def _compute_stability(accel_variance: float, f_t: float) -> float:
    """Normaliza varianza de aceleración a [0, 1].

    0.0 = perfectamente estable (señal constante o lineal).
    1.0 = altamente inestable.
    """
    normalizer = abs(f_t) if abs(f_t) > 1e-6 else 1.0
    return min(accel_variance / normalizer, 1.0)
