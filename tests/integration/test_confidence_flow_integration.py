"""Integration test for unified confidence calibration flow.

Tests the complete pipeline:
1. Raw confidence from engine fusion
2. Unified temperature‑scaled sigmoid calibration (infrastructure/ml/calibration/)
3. Floor=0.30, ceiling=0.95, data_quality adjusts temperature
"""

import pytest

from iot_machine_learning.infrastructure.ml.calibration import ConfidenceCalibrator


class TestConfidenceFlowIntegration:
    """Integration tests for unified confidence calibration flow."""

    def test_happy_path_full_flow(self):
        """Happy path: raw confidence → calibrate with good data quality."""
        raw_engine_confidences = [0.85, 0.80, 0.90]
        fused_confidence = sum(raw_engine_confidences) / len(raw_engine_confidences)

        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            score=fused_confidence,
            regime="STABLE",
            data_quality=1.0,
        )

        assert 0.30 <= result.calibrated <= 0.95
        assert result.raw == fused_confidence

    def test_edge_case_low_quality_data(self):
        """Low data_quality boosts temperature, lowering calibrated confidence."""
        calibrator = ConfidenceCalibrator()
        good = calibrator.calibrate(score=0.85, regime="STABLE", data_quality=1.0)
        bad = calibrator.calibrate(score=0.85, regime="STABLE", data_quality=0.2)

        assert bad.calibrated <= good.calibrated
        assert any("×1.6" in r for r in bad.reasons)
        assert 0.30 <= bad.calibrated <= 0.95

    def test_volatile_regime_lowers_confidence(self):
        """VOLATILE regime uses higher temperature → lower calibrated."""
        calibrator = ConfidenceCalibrator()
        stable = calibrator.calibrate(score=0.85, regime="STABLE", data_quality=1.0)
        volatile = calibrator.calibrate(score=0.85, regime="VOLATILE", data_quality=1.0)
        assert volatile.calibrated <= stable.calibrated

    def test_same_input_same_output(self):
        """Same input always produces same output (deterministic)."""
        calibrator = ConfidenceCalibrator()
        r1 = calibrator.calibrate(score=0.75, regime="STABLE", data_quality=1.0)
        r2 = calibrator.calibrate(score=0.75, regime="STABLE", data_quality=1.0)
        assert r1.calibrated == r2.calibrated
        assert r1.reasons == r2.reasons

    def test_floor_never_below_030(self):
        """Calibrated confidence never below 0.30."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(score=-10.0, regime="VOLATILE", data_quality=0.1)
        assert result.calibrated == 0.30

    def test_ceiling_never_above_095(self):
        """Calibrated confidence never above 0.95."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(score=500.0, regime="STABLE", data_quality=1.0)
        assert result.calibrated == 0.95

    def test_invalid_score_returns_floor(self):
        """NaN or inf score returns floor=0.30."""
        calibrator = ConfidenceCalibrator()
        for bad in [float("nan"), float("inf"), float("-inf")]:
            result = calibrator.calibrate(score=bad, regime="STABLE")
            assert result.calibrated == 0.30
