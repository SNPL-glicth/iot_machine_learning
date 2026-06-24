"""Tests for Fase 13: Confidence Calibration Unification.

Tests verify:
- CONFIDENCE singleton has regime temperatures
- TemperatureScaler uses CONFIDENCE singleton
- Unified ConfidenceCalibrator floor is 0.30, ceiling 0.95
"""

import pytest

from core.parameters.numerical_constants import CONFIDENCE
from core.tuning.temperature_scaling import TemperatureScaler
from iot_machine_learning.infrastructure.ml.calibration import ConfidenceCalibrator


class TestConfidenceSingletonRegimeTemps:
    """Tests for CONFIDENCE singleton regime temperature fields."""

    def test_confidence_has_temp_stable(self):
        assert hasattr(CONFIDENCE, "TEMP_STABLE")
        assert CONFIDENCE.TEMP_STABLE == 1.2

    def test_confidence_has_temp_volatile(self):
        assert hasattr(CONFIDENCE, "TEMP_VOLATILE")
        assert CONFIDENCE.TEMP_VOLATILE == 2.0

    def test_confidence_has_all_regime_temps(self):
        assert hasattr(CONFIDENCE, "TEMP_STABLE")
        assert hasattr(CONFIDENCE, "TEMP_TRENDING")
        assert hasattr(CONFIDENCE, "TEMP_VOLATILE")
        assert hasattr(CONFIDENCE, "TEMP_NOISY")
        assert hasattr(CONFIDENCE, "TEMP_DEFAULT")
        assert CONFIDENCE.TEMP_STABLE == 1.2
        assert CONFIDENCE.TEMP_TRENDING == 1.5
        assert CONFIDENCE.TEMP_VOLATILE == 2.0
        assert CONFIDENCE.TEMP_NOISY == 1.8
        assert CONFIDENCE.TEMP_DEFAULT == 1.5


class TestTemperatureScalerUsesConfidence:
    """Tests for TemperatureScaler using CONFIDENCE singleton."""

    def test_temperature_scaler_uses_confidence_floor(self):
        scaler = TemperatureScaler()
        assert scaler._floor == CONFIDENCE.MIN_CONFIDENCE

    def test_temperature_scaler_uses_confidence_ceiling(self):
        scaler = TemperatureScaler()
        assert scaler._ceiling == CONFIDENCE.MAX_CONFIDENCE
        assert scaler._ceiling == 0.95

    def test_temperature_scaler_regime_temps_from_singleton(self):
        scaler = TemperatureScaler()
        result = scaler.scale(confidence=0.8, regime="STABLE")
        assert result.temperature_used == CONFIDENCE.TEMP_STABLE
        assert result.temperature_used == 1.2

    def test_temperature_scaler_volatile_temp(self):
        scaler = TemperatureScaler()
        result = scaler.scale(confidence=0.8, regime="VOLATILE")
        assert result.temperature_used == CONFIDENCE.TEMP_VOLATILE
        assert result.temperature_used == 2.0


class TestUnifiedConfidenceCalibrator:
    """Tests for unified ConfidenceCalibrator."""

    def test_floor_is_030(self):
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(score=-10.0, regime="STABLE")
        assert result.calibrated == 0.30

    def test_ceiling_is_095(self):
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(score=500.0, regime="STABLE")
        assert result.calibrated == 0.95

    def test_data_quality_boosts_temperature(self):
        calibrator = ConfidenceCalibrator()
        good = calibrator.calibrate(score=0.85, regime="STABLE", data_quality=1.0)
        bad = calibrator.calibrate(score=0.85, regime="STABLE", data_quality=0.4)
        assert bad.calibrated <= good.calibrated
