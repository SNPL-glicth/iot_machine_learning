"""Tests for unified ConfidenceCalibrator (infrastructure/ml/calibration/).

The old additive‑penalty domain calibrator has been removed.
All calibration now uses temperature‑scaled sigmoid with:
  * floor=0.30, ceiling=0.95
  * data_quality boosts temperature
  * regime‑aware base temperature
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.calibration.confidence_calibrator import (
    CalibratedConfidence,
    ConfidenceCalibrator,
)


class TestCalibratorBasics:
    def test_calibrator_construction(self):
        calibrator = ConfidenceCalibrator()
        assert calibrator is not None

    def test_calibrated_confidence_dataclass(self):
        result = CalibratedConfidence(
            calibrated=0.65,
            raw=0.85,
            penalty_applied=0.20,
            reasons=["reason1", "reason2"],
        )
        assert result.calibrated == 0.65
        assert result.raw == 0.85
        assert result.penalty_applied == 0.20
        assert len(result.reasons) == 2


class TestUnifiedCalibrator:
    def test_floor_030(self):
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(score=-10.0, regime="STABLE")
        assert result.calibrated == 0.30

    def test_ceiling_095(self):
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(score=500.0, regime="STABLE")
        assert result.calibrated == 0.95

    def test_data_quality_below_05(self):
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(score=0.85, regime="STABLE", data_quality=0.4)
        assert any("×1.3" in r for r in result.reasons)

    def test_data_quality_below_03(self):
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(score=0.85, regime="STABLE", data_quality=0.2)
        assert any("×1.6" in r for r in result.reasons)

    def test_same_input_same_output(self):
        calibrator = ConfidenceCalibrator()
        r1 = calibrator.calibrate(score=0.75, regime="STABLE")
        r2 = calibrator.calibrate(score=0.75, regime="STABLE")
        assert r1.calibrated == r2.calibrated

    def test_regime_temperature_override(self):
        calibrator = ConfidenceCalibrator()
        stable = calibrator.calibrate(score=0.85, regime="STABLE")
        volatile = calibrator.calibrate(score=0.85, regime="VOLATILE")
        assert volatile.calibrated <= stable.calibrated

    def test_invalid_score_returns_floor(self):
        calibrator = ConfidenceCalibrator()
        for bad in [float("nan"), float("inf"), float("-inf")]:
            result = calibrator.calibrate(score=bad)
            assert result.calibrated == 0.30
