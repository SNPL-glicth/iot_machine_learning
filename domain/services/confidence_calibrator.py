"""Backward-compatible shim — now consolidated at infrastructure/ml/calibration/

Exposes the new unified ConfidenceCalibrator and CalibratedConfidence.
FLOOR=0.30, CEILING=0.95, uses temperature‑scaled sigmoid.
"""
from iot_machine_learning.infrastructure.ml.calibration.confidence_calibrator import (
    ConfidenceCalibrator as _C,
    CalibratedConfidence,
)
from iot_machine_learning.infrastructure.ml.calibration.confidence_calibrator import (
    CalibratedConfidence as _CalibratedConfidence,
)

# Re-export with same names
ConfidenceCalibrator = _C

__all__ = ["ConfidenceCalibrator", "CalibratedConfidence"]
