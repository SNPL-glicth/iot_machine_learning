"""Calibradores de confianza probabilística.

Implementaciones del ConfidenceCalibratorPort:
- PlattCalibrator: Platt Scaling con SGD online
- IsotonicCalibrator: Isotonic Regression con warm updates
- RegimeAwareCalibrator: Selección dinámica por régimen
"""

from .platt_calibrator import PlattCalibrator
from .isotonic_calibrator import IsotonicCalibrator
from .regime_aware_calibrator import RegimeAwareCalibrator

__all__ = ["PlattCalibrator", "IsotonicCalibrator", "RegimeAwareCalibrator"]
