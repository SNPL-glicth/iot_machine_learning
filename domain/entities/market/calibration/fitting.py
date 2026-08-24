"""FASE 10.1 — Calibration Fitting (Platt, Bucket)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Final

from .context_types import CalibrationMethod


def fit_platt_scaling(
    raw_probs: list[float],
    outcomes: list[bool],  # True = up, False = down
) -> tuple[float, float]:
    """Ajusta parámetros Platt scaling (a, b) usando maximum likelihood.
    
    Implementación simplificada usando gradient descent básico.
    """
    if len(raw_probs) != len(outcomes):
        raise ValueError("raw_probs y outcomes deben tener mismo tamaño")
    
    if len(raw_probs) < 20:
        # No hay suficientes datos: retornar identidad (a=1, b=0)
        return 1.0, 0.0
    
    # Inicializar parámetros (a=1, b=0 = identidad)
    a, b = 1.0, 0.0
    learning_rate = 0.1
    iterations = 100
    
    for _ in range(iterations):
        gradient_a = 0.0
        gradient_b = 0.0
        
        for prob, outcome in zip(raw_probs, outcomes):
            y = 1.0 if outcome else 0.0
            x = a * prob + b
            sigma = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
            
            gradient_a += (sigma - y) * prob
            gradient_b += (sigma - y)
        
        # Normalizar gradientes
        n = len(raw_probs)
        gradient_a /= n
        gradient_b /= n
        
        # Actualizar parámetros
        a -= learning_rate * gradient_a
        b -= learning_rate * gradient_b
    
    return a, b


def fit_bucket_calibration(
    raw_probs: list[float],
    outcomes: list[bool],
) -> dict[float, float]:
    """Ajusta calibración por bucket (como el observatory)."""
    if len(raw_probs) != len(outcomes):
        raise ValueError("raw_probs y outcomes deben tener mismo tamaño")
    
    if len(raw_probs) < 20:
        return {}  # No hay suficientes datos
    
    # Agrupar por bucket
    buckets: dict[float, list[tuple[float, bool]]] = defaultdict(list)
    for prob, outcome in zip(raw_probs, outcomes):
        bucket = round(prob * 10) / 10
        buckets[bucket].append((prob, outcome))
    
    # Calcular tasa de acierto por bucket
    calibration: dict[float, float] = {}
    for bucket, items in buckets.items():
        if len(items) < 5:  # Mínimo 5 muestras por bucket
            continue
        hit_rate = sum(1 for _, outcome in items if outcome) / len(items)
        calibration[bucket] = hit_rate
    
    return calibration