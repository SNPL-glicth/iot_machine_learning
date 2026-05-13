"""Tests for core/temperature_scaling.py."""

import pytest

from core.tuning.temperature_scaling import TemperatureScaler, TemperatureScalingResult


class TestTemperatureScaler:
    def test_init_default_temperature(self):
        scaler = TemperatureScaler()
        assert scaler._default_temperature == 1.5
        assert scaler._floor == 0.3  # CONFIDENCE.MIN_CONFIDENCE
        assert scaler._ceiling == 0.95

    def test_scale_stable_regime(self):
        scaler = TemperatureScaler()
        result = scaler.scale(confidence=0.8, regime="STABLE")
        assert result.temperature_used == 1.2
        assert result.regime == "STABLE"
        assert 0.3 <= result.scaled_confidence <= 0.95  # Floor now 0.3

    def test_scale_volatile_regime(self):
        scaler = TemperatureScaler()
        result = scaler.scale(confidence=0.8, regime="VOLATILE")
        assert result.temperature_used == 2.0
        assert result.regime == "VOLATILE"
        # Higher temperature should reduce confidence
        assert result.scaled_confidence < 0.8

    def test_scale_preserves_floor_ceiling(self):
        scaler = TemperatureScaler(floor=0.1, ceiling=0.9)
        result_low = scaler.scale(confidence=0.01, regime="STABLE")
        result_high = scaler.scale(confidence=0.99, regime="STABLE")
        assert result_low.scaled_confidence >= 0.1
        assert result_high.scaled_confidence <= 0.9

    def test_temperature_override(self):
        scaler = TemperatureScaler()
        result = scaler.scale(confidence=0.7, regime="STABLE", temperature_override=2.5)
        assert result.temperature_used == 2.5

    def test_calibrate_temperature_converges(self):
        scaler = TemperatureScaler()
        confidences = [0.6, 0.7, 0.8, 0.9]
        temperature = scaler.calibrate_temperature(confidences, target_mean=0.5)
        assert 0.5 <= temperature <= 3.0

    def test_formula_documented_in_result(self):
        scaler = TemperatureScaler()
        result = scaler.scale(confidence=0.75, regime="DEFAULT")
        assert "sigmoid" in result.formula
        assert "0.75" in result.formula or "0.750" in result.formula

    def test_unknown_regime_uses_default(self):
        scaler = TemperatureScaler(default_temperature=1.8)
        result = scaler.scale(confidence=0.7, regime="UNKNOWN_REGIME")
        assert result.temperature_used == 1.8
