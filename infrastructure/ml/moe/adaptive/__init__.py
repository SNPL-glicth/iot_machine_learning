"""Módulo de Adaptive Meta-Gating y Conformal Risk Management (Fase 2).

Proporciona calibración online mediante e-values ponderados por Hedge y compuertas
de decisión multiobjetivo certificadas bajo la desigualdad maximal de Ville.
"""

from __future__ import annotations

from .conformal_calibrator import OnlineConformalCalibrator
from .dynamic_gate import LatencyBudgetAwareGate

__all__ = [
    "OnlineConformalCalibrator",
    "LatencyBudgetAwareGate",
]
