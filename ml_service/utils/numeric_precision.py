"""Funciones canónicas de precisión numérica.

FASE 2: Modelo matemático unificado para evitar desfases entre servicios.

Política de precisión:
- BD: DECIMAL(15,5) — 5 decimales exactos
- Cálculos internos: Python float (IEEE 754 double) — ~15 dígitos significativos
- Redondeo: SOLO en frontera (UI) — nunca en cálculos intermedios
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

# Precisión canónica para valores de sensor (coincide con DECIMAL(15,5) de SQL Server)
SENSOR_VALUE_PRECISION = 5

# Precisión para cálculos de delta
DELTA_PRECISION = 6


def safe_float(value, default: float = 0.0) -> float:
    """Convierte un valor a float con validación de NaN/Infinity.
    
    FASE 4: Guard para evitar datos corruptos en cálculos.
    
    Args:
        value: Valor a convertir (puede ser None, str, Decimal, etc.)
        default: Valor por defecto si es inválido
    
    Returns:
        Float válido o default si el valor es None, NaN o Infinity
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


def is_valid_sensor_value(value) -> bool:
    """Verifica si un valor es válido para un sensor.
    
    Args:
        value: Valor a verificar
    
    Returns:
        True si es un número finito válido
    """
    if value is None:
        return False
    try:
        f = float(value)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def filter_valid_values(values: List) -> List[float]:
    """Filtra una lista dejando solo valores válidos (finitos).
    
    Args:
        values: Lista de valores potencialmente inválidos
    
    Returns:
        Lista de floats válidos (sin None, NaN, Infinity)
    """
    return [safe_float(v) for v in values if is_valid_sensor_value(v)]


def round_for_display(value: float, decimals: int = 2) -> float:
    """Redondea un valor a la precisión canónica.
    
    USAR SOLO para display en UI, nunca para cálculos intermedios.
    """
    if not math.isfinite(value):
        return value
    factor = 10 ** decimals
    return round(value * factor) / factor


def round_for_persistence(value: float) -> float:
    """Redondea para persistencia (coincide con DECIMAL(15,5))."""
    if not math.isfinite(value):
        return value
    factor = 10 ** SENSOR_VALUE_PRECISION
    return round(value * factor) / factor


def compute_delta(current: float, previous: float) -> float:
    """Calcula delta entre dos valores con precisión canónica.
    
    Args:
        current: Valor actual
        previous: Valor anterior
    
    Returns:
        Delta con precisión DELTA_PRECISION
    """
    if not math.isfinite(current) or not math.isfinite(previous):
        return 0.0
    delta = current - previous
    # Redondear a precisión delta para evitar errores de punto flotante
    factor = 10 ** DELTA_PRECISION
    return round(delta * factor) / factor


def compute_abs_delta(current: float, previous: float) -> float:
    """Calcula delta absoluto."""
    return abs(compute_delta(current, previous))


def compute_relative_delta(current: float, previous: float) -> float:
    """Calcula delta relativo (porcentaje de cambio).
    
    Args:
        current: Valor actual
        previous: Valor anterior
    
    Returns:
        Delta relativo (0.1 = 10%)
    """
    if not math.isfinite(current) or not math.isfinite(previous):
        return 0.0
    if abs(previous) < 1e-10:
        # Evitar división por cero
        return 0.0
    delta = current - previous
    return delta / abs(previous)


def compute_simple_moving_average(values: List[float], window: Optional[int] = None) -> float:
    """Calcula promedio móvil simple (SMA).
    
    Args:
        values: Lista de valores
        window: Tamaño de ventana (default: todos los valores)
    
    Returns:
        Promedio móvil
    """
    if not values:
        return 0.0
    
    effective_window = min(window, len(values)) if window else len(values)
    slice_values = values[-effective_window:]
    
    finite_values = [v for v in slice_values if math.isfinite(v)]
    if not finite_values:
        return 0.0
    
    return sum(finite_values) / len(finite_values)


def compute_exponential_moving_average(values: List[float], alpha: float = 0.2) -> float:
    """Calcula promedio móvil exponencial (EMA).
    
    Args:
        values: Lista de valores
        alpha: Factor de suavizado (0 < alpha <= 1). Mayor alpha = más peso al valor reciente.
    
    Returns:
        EMA del último valor
    """
    if not values:
        return 0.0
    if alpha <= 0 or alpha > 1:
        alpha = 0.2
    
    ema = values[0] if math.isfinite(values[0]) else 0.0
    for i in range(1, len(values)):
        v = values[i] if math.isfinite(values[i]) else ema
        ema = alpha * v + (1 - alpha) * ema
    
    return ema


def apply_median_filter(values: List[float], window: int = 3) -> List[float]:
    """Aplica filtro de mediana para eliminar outliers.
    
    Args:
        values: Lista de valores
        window: Tamaño de ventana (debe ser impar, default: 3)
    
    Returns:
        Lista filtrada
    """
    if len(values) < window:
        return values.copy()
    
    # Asegurar ventana impar
    if window % 2 == 0:
        window += 1
    
    half_window = window // 2
    result: List[float] = []
    
    for i in range(len(values)):
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)
        window_values = sorted(values[start:end])
        median = window_values[len(window_values) // 2]
        result.append(median)
    
    return result


def compute_window_stats(values: List[float]) -> Tuple[
    Optional[float], Optional[float], Optional[float], Optional[float]
]:
    """Calcula estadísticas de ventana (min, max, delta, mean).
    
    Returns:
        Tupla (min, max, delta, mean)
    """
    finite_values = [v for v in values if math.isfinite(v)]
    if not finite_values:
        return (None, None, None, None)
    
    min_val = min(finite_values)
    max_val = max(finite_values)
    delta = compute_delta(max_val, min_val)
    mean = sum(finite_values) / len(finite_values)
    
    return (min_val, max_val, delta, mean)


def is_spike(
    current_value: float,
    previous_value: float,
    abs_delta_threshold: Optional[float] = None,
    rel_delta_threshold: Optional[float] = None,
) -> bool:
    """Detecta si un valor es un spike basado en umbrales de delta."""
    if abs_delta_threshold is not None:
        abs_delta = compute_abs_delta(current_value, previous_value)
        if abs_delta >= abs_delta_threshold:
            return True
    
    if rel_delta_threshold is not None:
        rel_delta = abs(compute_relative_delta(current_value, previous_value))
        if rel_delta >= rel_delta_threshold:
            return True
    
    return False
