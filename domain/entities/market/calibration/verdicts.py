"""FASE 10.5 — Calibration Verdicts and Fallback Levels."""

from __future__ import annotations

from enum import Enum


class CalibrationVerdict(Enum):
    """Veredicto de evaluación de calibrador."""
    
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INSUFFICIENT_DATA = "insufficient_data"


class FallbackLevel(Enum):
    """Niveles de fallback para calibración."""

    CONTEXT = "context"  # strategy·horizon·regime
    HORIZON = "horizon"  # strategy·horizon
    REGIME = "regime"  # strategy·regime
    STRATEGY = "strategy"  # strategy
    GLOBAL = "global"  # todos los datos
    UNAVAILABLE = "unavailable"  # sin calibración