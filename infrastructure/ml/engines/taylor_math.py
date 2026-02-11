"""Funciones matemáticas puras para Series de Taylor con diferencias finitas.

Extraído de ml/core/taylor_predictor.py.
Responsabilidad ÚNICA: cálculos numéricos de derivadas, expansión Taylor
y varianza de aceleración. Sin I/O, sin estado, sin logging.

Fórmulas:
    f'(t)   ≈ (f(t) - f(t-1)) / Δt                          (velocidad)
    f''(t)  ≈ (f(t) - 2·f(t-1) + f(t-2)) / Δt²              (aceleración)
    f'''(t) ≈ (f(t) - 3·f(t-1) + 3·f(t-2) - f(t-3)) / Δt³   (jerk)
    f(t+h)  ≈ f(t) + f'(t)·h + f''(t)·h²/2! + f'''(t)·h³/3!
"""

from __future__ import annotations

from typing import Dict, List

# Mínimo de puntos para calcular varianza de aceleración
MIN_ACCEL_HISTORY: int = 4


def compute_finite_differences(
    values: List[float],
    dt: float,
    order: int,
) -> Dict[str, float]:
    """Calcula derivadas por diferencias finitas hacia atrás.

    Args:
        values: Serie temporal (más reciente al final).
        dt: Paso temporal (Δt) entre muestras consecutivas.
        order: Orden máximo de derivada a calcular (1–3).

    Returns:
        Dict con ``f_t``, ``f_prime``, ``f_double_prime``, ``f_triple_prime``.
    """
    n = len(values)
    f_t = values[-1]

    f_prime = 0.0
    if order >= 1 and n >= 2:
        f_prime = (values[-1] - values[-2]) / dt

    f_double_prime = 0.0
    if order >= 2 and n >= 3:
        f_double_prime = (values[-1] - 2.0 * values[-2] + values[-3]) / (dt * dt)

    f_triple_prime = 0.0
    if order >= 3 and n >= 4:
        f_triple_prime = (
            values[-1]
            - 3.0 * values[-2]
            + 3.0 * values[-3]
            - values[-4]
        ) / (dt * dt * dt)

    return {
        "f_t": f_t,
        "f_prime": f_prime,
        "f_double_prime": f_double_prime,
        "f_triple_prime": f_triple_prime,
    }


def taylor_expand(
    derivs: Dict[str, float],
    h: float,
    order: int,
) -> float:
    """Evalúa la expansión de Taylor: f(t+h).

    Args:
        derivs: Diccionario de derivadas (output de ``compute_finite_differences``).
        h: Horizonte de predicción (en unidades de Δt).
        order: Orden de la expansión (1–3).

    Returns:
        Valor predicho f(t+h).
    """
    result = derivs["f_t"]

    if order >= 1:
        result += derivs["f_prime"] * h
    if order >= 2:
        result += derivs["f_double_prime"] * (h * h) / 2.0
    if order >= 3:
        result += derivs["f_triple_prime"] * (h * h * h) / 6.0

    return result


def compute_accel_variance(values: List[float], dt: float) -> float:
    """Calcula la varianza de la aceleración (f'') sobre los últimos puntos.

    Se usa para estimar la estabilidad de la serie.

    Args:
        values: Serie temporal.
        dt: Paso temporal.

    Returns:
        Varianza de f''. Retorna 0.0 si no hay suficientes puntos.
    """
    n = len(values)
    if n < MIN_ACCEL_HISTORY:
        return 0.0

    accels: List[float] = []
    dt_sq = dt * dt
    for i in range(2, n):
        accel = (values[i] - 2.0 * values[i - 1] + values[i - 2]) / dt_sq
        accels.append(accel)

    if len(accels) < 2:
        return 0.0

    mean_accel = sum(accels) / len(accels)
    variance = sum((a - mean_accel) ** 2 for a in accels) / len(accels)
    return variance


def compute_dt(timestamps: list[float] | None) -> float:
    """Calcula Δt a partir de timestamps o usa default 1.0.

    Usa la mediana de las diferencias consecutivas para robustez.

    Args:
        timestamps: Timestamps Unix opcionales.

    Returns:
        Δt > 0. Mínimo 1e-6.
    """
    if timestamps is None or len(timestamps) < 2:
        return 1.0

    diffs = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
        if timestamps[i] > timestamps[i - 1]
    ]

    if not diffs:
        return 1.0

    diffs_sorted = sorted(diffs)
    mid = len(diffs_sorted) // 2
    if len(diffs_sorted) % 2 == 0:
        dt = (diffs_sorted[mid - 1] + diffs_sorted[mid]) / 2.0
    else:
        dt = diffs_sorted[mid]

    return max(dt, 1e-6)
