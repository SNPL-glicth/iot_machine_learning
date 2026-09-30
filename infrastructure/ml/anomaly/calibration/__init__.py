"""Módulo de Calibración Adaptativa de Anomalías de ZENIN.

Desacopla la interpretación probabilística/calibrada de la lógica de los detectores base.
"""
from __future__ import annotations

from .guardrail import CalibrationGuardrail, CalibrationState
from .layer import AdaptiveDetectorCalibrationLayer
from .profile import DetectorCalibrationProfile, compute_calibrated_score

__all__ = [
    "CalibrationState",
    "CalibrationGuardrail",
    "DetectorCalibrationProfile",
    "compute_calibrated_score",
    "AdaptiveDetectorCalibrationLayer",
]
