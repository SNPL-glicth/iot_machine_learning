"""FASE 10.1 — Factory functions for Context Calibration."""

from __future__ import annotations

from .context_types import CalibrationMethod
from .context_calibrator import ContextCalibrator


def fit_context_calibrator(
    data: list[tuple["ContextKey", float, bool]],
    method: CalibrationMethod = CalibrationMethod.PLATT,
) -> ContextCalibrator:
    """Factory para crear y ajustar un calibrador contextual."""
    from .context_types import ContextKey
    calibrator = ContextCalibrator(method=method)
    calibrator.fit(data)
    return calibrator


def apply_calibration(
    calibrator: ContextCalibrator,
    context: "ContextKey",
    prob_raw: float,
) -> float:
    """Wrapper simple para aplicar calibración."""
    result = calibrator.calibrate(context, prob_raw)
    return result.prob_calibrated