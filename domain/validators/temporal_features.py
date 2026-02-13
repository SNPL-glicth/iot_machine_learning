"""Cómputo de features temporales para series de tiempo.

Responsabilidad ÚNICA: calcular velocidad, aceleración, jitter y
estadísticas agregadas a partir de valores + timestamps.

Función pura — sin I/O, sin estado, sin dependencias externas.
Usado por TimeSeries.temporal_features y SensorWindow.temporal_features.
"""

from __future__ import annotations

import math
from typing import List

from ..entities.temporal_features import TemporalFeatures


def compute_temporal_features(
    values: List[float],
    timestamps: List[float],
) -> TemporalFeatures:
    """Computa features temporales de una serie.

    Calcula velocidades (dv/dt), aceleraciones (d²v/dt²), jitter
    de muestreo, y estadísticas agregadas.

    Args:
        values: Valores numéricos ordenados cronológicamente.
        timestamps: Timestamps correspondientes (misma longitud).

    Returns:
        ``TemporalFeatures`` con todas las métricas derivadas.

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
        return TemporalFeatures(n_points=n)

    # --- Δt values ---
    dt_values = _compute_dt_values(timestamps)

    # --- Velocities: dv/dt ---
    velocities = _compute_velocities(values, dt_values)

    # --- Accelerations: d²v/dt² ---
    accelerations = _compute_accelerations(velocities, dt_values)

    # --- Jitter (CV of Δt) ---
    jitter = _compute_jitter(dt_values)

    # --- Aggregate stats ---
    mean_dt = sum(dt_values) / len(dt_values) if dt_values else 1.0

    mean_vel, std_vel, max_abs_vel = _stats(velocities)
    mean_acc, std_acc, max_abs_acc = _stats(accelerations)

    last_velocity = velocities[-1] if velocities else 0.0
    last_acceleration = accelerations[-1] if accelerations else 0.0

    return TemporalFeatures(
        velocities=velocities,
        accelerations=accelerations,
        dt_values=dt_values,
        mean_velocity=mean_vel,
        std_velocity=std_vel,
        max_abs_velocity=max_abs_vel,
        mean_acceleration=mean_acc,
        std_acceleration=std_acc,
        max_abs_acceleration=max_abs_acc,
        jitter=jitter,
        mean_dt=mean_dt,
        last_velocity=last_velocity,
        last_acceleration=last_acceleration,
        n_points=n,
    )


def _compute_dt_values(timestamps: List[float]) -> List[float]:
    """Calcula intervalos Δt entre puntos consecutivos.

    Si Δt ≤ 0 (timestamps no monótonos o duplicados), se reemplaza
    por un epsilon para evitar divisiones por cero.
    """
    eps = 1e-9
    dt_values: List[float] = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        dt_values.append(dt if dt > eps else eps)
    return dt_values


def _compute_velocities(
    values: List[float],
    dt_values: List[float],
) -> List[float]:
    """Calcula dv/dt para cada par consecutivo."""
    velocities: List[float] = []
    for i in range(len(dt_values)):
        dv = values[i + 1] - values[i]
        velocities.append(dv / dt_values[i])
    return velocities


def _compute_accelerations(
    velocities: List[float],
    dt_values: List[float],
) -> List[float]:
    """Calcula d²v/dt² a partir de velocidades consecutivas.

    La aceleración en el punto i+1 se calcula como:
        a[i] = (v[i+1] - v[i]) / dt_mid
    donde dt_mid es el promedio de dt[i] y dt[i+1] (centrado).
    """
    if len(velocities) < 2:
        return []

    accelerations: List[float] = []
    for i in range(len(velocities) - 1):
        # dt_mid: promedio de los dos intervalos adyacentes
        dt_mid = (dt_values[i] + dt_values[i + 1]) / 2.0 if (i + 1) < len(dt_values) else dt_values[i]
        dv = velocities[i + 1] - velocities[i]
        accelerations.append(dv / dt_mid)
    return accelerations


def _compute_jitter(dt_values: List[float]) -> float:
    """Calcula jitter como coeficiente de variación de Δt."""
    if len(dt_values) < 2:
        return 0.0

    mean_dt = sum(dt_values) / len(dt_values)
    if mean_dt < 1e-12:
        return 0.0

    variance = sum((dt - mean_dt) ** 2 for dt in dt_values) / (len(dt_values) - 1)
    std_dt = math.sqrt(variance)
    return std_dt / mean_dt


def _stats(values: List[float]) -> tuple:
    """Calcula (mean_abs, std, max_abs) de una lista de valores.

    Returns:
        Tupla (mean_abs, std, max_abs). Retorna (0, 0, 0) si vacía.
    """
    if not values:
        return 0.0, 0.0, 0.0

    n = len(values)
    abs_values = [abs(v) for v in values]
    mean_abs = sum(abs_values) / n
    max_abs = max(abs_values)

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    std = math.sqrt(variance)

    return mean_abs, std, max_abs
