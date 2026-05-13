"""Tests for Fase 13: Confidence Calibration Unification.

Tests verify:
- CONFIDENCE singleton has regime temperatures
- TemperatureScaler uses CONFIDENCE singleton
- domain ConfidenceCalibrator floor is 0.05 (intentional)
"""

import pytest

from core.parameters.numerical_constants import CONFIDENCE
from core.tuning.temperature_scaling import TemperatureScaler
from domain.services.confidence_calibrator import ConfidenceCalibrator


class TestConfidenceSingletonRegimeTemps:
    """Tests for CONFIDENCE singleton regime temperature fields."""

    def test_confidence_has_temp_stable(self):
        """CONFIDENCE.TEMP_STABLE == 1.2"""
        assert hasattr(CONFIDENCE, "TEMP_STABLE")
        assert CONFIDENCE.TEMP_STABLE == 1.2

    def test_confidence_has_temp_volatile(self):
        """CONFIDENCE.TEMP_VOLATILE == 2.0"""
        assert hasattr(CONFIDENCE, "TEMP_VOLATILE")
        assert CONFIDENCE.TEMP_VOLATILE == 2.0

    def test_confidence_has_all_regime_temps(self):
        """All 5 regime temperatures present."""
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
        """Default floor == CONFIDENCE.MIN_CONFIDENCE (0.3)."""
        scaler = TemperatureScaler()
        assert scaler._floor == CONFIDENCE.MIN_CONFIDENCE
        assert scaler._floor == 0.3

    def test_temperature_scaler_uses_confidence_ceiling(self):
        """Default ceiling == CONFIDENCE.MAX_CONFIDENCE (0.95)."""
        scaler = TemperatureScaler()
        assert scaler._ceiling == CONFIDENCE.MAX_CONFIDENCE
        assert scaler._ceiling == 0.95

    def test_temperature_scaler_regime_temps_from_singleton(self):
        """STABLE regime uses CONFIDENCE.TEMP_STABLE."""
        scaler = TemperatureScaler()
        result = scaler.scale(confidence=0.8, regime="STABLE")
        assert result.temperature_used == CONFIDENCE.TEMP_STABLE
        assert result.temperature_used == 1.2

    def test_temperature_scaler_volatile_temp(self):
        """VOLATILE regime uses CONFIDENCE.TEMP_VOLATILE (2.0)."""
        scaler = TemperatureScaler()
        result = scaler.scale(confidence=0.8, regime="VOLATILE")
        assert result.temperature_used == CONFIDENCE.TEMP_VOLATILE
        assert result.temperature_used == 2.0


class TestDomainConfidenceCalibrator:
    """Tests for domain ConfidenceCalibrator floor documentation."""

    def test_penalty_calibrator_floor_is_005(self):
        """CONFIDENCE_FLOOR == 0.05 (intentional, not global MIN)."""
        calibrator = ConfidenceCalibrator()
        assert calibrator.CONFIDENCE_FLOOR == 0.05

    def test_penalty_calibrator_floor_different_from_global(self):
        """Floor 0.05 != CONFIDENCE.MIN_CONFIDENCE (0.3)."""
        calibrator = ConfidenceCalibrator()
        assert calibrator.CONFIDENCE_FLOOR != CONFIDENCE.MIN_CONFIDENCE
        assert calibrator.CONFIDENCE_FLOOR == 0.05
        assert CONFIDENCE.MIN_CONFIDENCE == 0.3

    def test_penalty_calibrator_has_docstring(self):
        """ConfidenceCalibrator class has docstring explaining difference."""
        assert ConfidenceCalibrator.__doc__ is not None
        assert "Penalty-Based Confidence Calibrator" in ConfidenceCalibrator.__doc__
        assert "temperature_scaling.py" in ConfidenceCalibrator.__doc__
        assert "COMPLEMENTARIOS" in ConfidenceCalibrator.__doc__
