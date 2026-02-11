"""Validaciones numéricas reutilizables para UTSAE.

Migrado desde ml/core/validators.py → domain/validators/numeric.py
Responsabilidad ÚNICA: guards contra NaN, Inf, ventanas insuficientes
y predicciones divergentes. Sin I/O, sin dependencias externas.

Usado por TaylorPredictionEngine, KalmanSignalFilter y cualquier motor futuro.
"""

from __future__ import annotations

import math
from typing import List, Tuple


class ValidationError(ValueError):
    """Error de validación de datos numéricos para motores UTSAE.

    Hereda de ``ValueError`` para compatibilidad con handlers existentes
    que capturan ``ValueError`` en el pipeline de predicción.
    """

    pass


def validate_window(values: List[float], min_size: int = 1) -> None:
    """Valida que una ventana de valores sea utilizable por un motor.

    Checks:
    1. La lista no está vacía y tiene al menos ``min_size`` elementos.
    2. No contiene ``NaN``.
    3. No contiene ``Inf`` / ``-Inf``.

    Args:
        values: Ventana de valores a validar.
        min_size: Mínimo de puntos requeridos.

    Raises:
        ValidationError: Si alguna condición no se cumple.
    """
    if not values:
        raise ValidationError(
            f"Se requieren al menos {min_size} valores, recibidos 0"
        )

    if len(values) < min_size:
        raise ValidationError(
            f"Se requieren al menos {min_size} valores, recibidos {len(values)}"
        )

    for i, v in enumerate(values):
        if not isinstance(v, (int, float)):
            raise ValidationError(
                f"Valor en posición {i} no es numérico: {type(v).__name__}"
            )
        if math.isnan(v):
            raise ValidationError(
                f"La ventana contiene valores NaN en posición {i}"
            )
        if math.isinf(v):
            raise ValidationError(
                f"La ventana contiene valores infinitos en posición {i}"
            )


def clamp_prediction(
    predicted: float,
    values: List[float],
    margin_pct: float = 0.3,
) -> Tuple[float, bool]:
    """Clampea una predicción al rango observado + margen para estabilidad.

    Args:
        predicted: Valor predicho (potencialmente divergente).
        values: Serie histórica observada.
        margin_pct: Fracción del rango permitida como margen (default 0.3).

    Returns:
        Tupla ``(clamped_value, was_clamped)``.

    Raises:
        ValidationError: Si ``values`` está vacía.
    """
    if not values:
        raise ValidationError("No se puede clampear sin valores de referencia")

    series_min = float(min(values))
    series_max = float(max(values))

    range_span = series_max - series_min
    margin = range_span * margin_pct if range_span > 0 else abs(series_min) * margin_pct

    if margin < 1e-9:
        margin = 0.1

    lower_bound = series_min - margin
    upper_bound = series_max + margin

    clamped = max(lower_bound, min(predicted, upper_bound))
    was_clamped = abs(clamped - predicted) > 1e-12

    return clamped, was_clamped


def safe_float(value: object, default: float = 0.0) -> float:
    """Convierte un valor a float con validación de NaN/Infinity.

    Args:
        value: Valor a convertir.
        default: Valor por defecto si es inválido.

    Returns:
        Float válido o ``default``.
    """
    if value is None:
        return default
    try:
        f = float(value)
        if not math.isfinite(f):
            return default
        return f
    except (TypeError, ValueError):
        return default
