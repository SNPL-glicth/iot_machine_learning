"""Filtros de señal para infraestructura ML.

Módulos:
- ``kalman_filter``    — Kalman 1D con auto-calibración de R y Q adaptativo.
- ``kalman_math``      — Funciones matemáticas puras de Kalman.
- ``ema_filter``       — EMA fijo y adaptativo (basado en innovación).
- ``median_filter``    — Mediana con ventana deslizante (robusto a spikes).
- ``filter_chain``     — Pipeline composable de filtros.
- ``filter_diagnostic`` — Métricas de calidad de filtrado.
"""

from .kalman_filter import KalmanSignalFilter
from .kalman_math import KalmanState, WarmupBuffer
from .kalman_adapter import KalmanFilterAdapter
from .ema_filter import EMASignalFilter, AdaptiveEMASignalFilter
from .median_filter import MedianSignalFilter
from .filter_chain import FilterChain
from .filter_diagnostic import FilterDiagnostic, compute_filter_diagnostic

__all__ = [
    "KalmanSignalFilter",
    "KalmanState",
    "WarmupBuffer",
    "KalmanFilterAdapter",
    "EMASignalFilter",
    "AdaptiveEMASignalFilter",
    "MedianSignalFilter",
    "FilterChain",
    "FilterDiagnostic",
    "compute_filter_diagnostic",
]
