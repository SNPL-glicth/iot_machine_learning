"""Unit tests for confidence calibration system.

Tests temperature-scaled sigmoid calibration with regime-aware adjustment.
"""

import math
import pytest

from iot_machine_learning.infrastructure.ml.calibration import ConfidenceCalibrator


class TestConfidenceCalibrator:
    """Test ConfidenceCalibrator class."""
    
    def test_sigmoid_output_range(self):
        """Calibrated output should always be in [0, 1]."""
        calibrator = ConfidenceCalibrator(temperature=1.5)
        
        test_scores = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
        
        for score in test_scores:
            calibrated = calibrator.calibrate(score)
            assert 0.0 <= calibrated <= 1.0, f"Score {score} → {calibrated} out of range"
    
    def test_temperature_scaling_effect(self):
        """Higher temperature should produce lower confidence for same score."""
        score = 2.0
        
        calibrator_low_temp = ConfidenceCalibrator(temperature=1.0)
        calibrator_high_temp = ConfidenceCalibrator(temperature=3.0)
        
        conf_low = calibrator_low_temp.calibrate(score)
        conf_high = calibrator_high_temp.calibrate(score)
        
        # Higher temperature → more conservative → lower confidence
        assert conf_high < conf_low
    
    def test_regime_adjustment(self):
        """Regime should affect temperature selection."""
        calibrator = ConfidenceCalibrator(
            temperature=1.5,
            regime_temperatures={
                "VOLATILE": 2.0,
                "STABLE": 1.2,
            }
        )
        
        score = 2.0
        
        conf_volatile = calibrator.calibrate(score, regime="VOLATILE")
        conf_stable = calibrator.calibrate(score, regime="STABLE")
        conf_default = calibrator.calibrate(score, regime=None)
        
        # VOLATILE uses higher temp → lower confidence
        assert conf_volatile < conf_default
        # STABLE uses lower temp → higher confidence
        assert conf_stable > conf_default
    
    def test_calibration_monotonicity(self):
        """Higher score should produce higher confidence."""
        calibrator = ConfidenceCalibrator(temperature=1.5)
        
        scores = [0.5, 1.0, 2.0, 3.0, 5.0]
        confidences = [calibrator.calibrate(s) for s in scores]
        
        # Check monotonicity
        for i in range(len(confidences) - 1):
            assert confidences[i] < confidences[i + 1], \
                f"Non-monotonic: {confidences[i]} >= {confidences[i + 1]}"
    
    def test_score_zero_returns_half(self):
        """Score of 0 should return approximately 0.5 (sigmoid property)."""
        calibrator = ConfidenceCalibrator(temperature=1.5)
        
        calibrated = calibrator.calibrate(0.0)
        
        # sigmoid(0) = 0.5
        assert abs(calibrated - 0.5) < 0.01
    
    def test_large_score_saturates_near_one(self):
        """Very large score should saturate near 1.0."""
        calibrator = ConfidenceCalibrator(temperature=1.5)
        
        calibrated = calibrator.calibrate(100.0)
        
        assert calibrated > 0.99
    
    def test_negative_score_returns_low_confidence(self):
        """Negative score should return low confidence."""
        calibrator = ConfidenceCalibrator(temperature=1.5)
        
        calibrated = calibrator.calibrate(-5.0)
        
        assert calibrated < 0.1
    
    def test_nan_score_returns_zero(self):
        """NaN score should return 0.0 with warning."""
        calibrator = ConfidenceCalibrator(temperature=1.5)
        
        calibrated = calibrator.calibrate(float('nan'))
        
        assert calibrated == 0.0
    
    def test_inf_score_returns_zero(self):
        """Inf score should return 0.0 with warning."""
        calibrator = ConfidenceCalibrator(temperature=1.5)
        
        calibrated = calibrator.calibrate(float('inf'))
        
        assert calibrated == 0.0
    
    def test_invalid_temperature_raises_error(self):
        """Temperature <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="temperature must be > 0"):
            ConfidenceCalibrator(temperature=0.0)
        
        with pytest.raises(ValueError, match="temperature must be > 0"):
            ConfidenceCalibrator(temperature=-1.0)
    
    def test_regime_not_in_dict_uses_default(self):
        """Unknown regime should use default temperature."""
        calibrator = ConfidenceCalibrator(
            temperature=1.5,
            regime_temperatures={"VOLATILE": 2.0}
        )
        
        score = 2.0
        
        conf_unknown = calibrator.calibrate(score, regime="UNKNOWN")
        conf_default = calibrator.calibrate(score, regime=None)
        
        # Should be identical (both use default temperature)
        assert abs(conf_unknown - conf_default) < 0.001
    
    def test_base_temperature_property(self):
        """Should expose base_temperature property."""
        calibrator = ConfidenceCalibrator(temperature=1.8)
        
        assert calibrator.base_temperature == 1.8
    
    def test_regime_temperatures_property(self):
        """Should expose regime_temperatures property."""
        regime_temps = {"VOLATILE": 2.0, "STABLE": 1.2}
        calibrator = ConfidenceCalibrator(
            temperature=1.5,
            regime_temperatures=regime_temps
        )
        
        assert calibrator.regime_temperatures == regime_temps
    
    def test_numerical_stability_extreme_values(self):
        """Should handle extreme values without overflow."""
        calibrator = ConfidenceCalibrator(temperature=1.0)
        
        # Very large positive
        calibrated_large = calibrator.calibrate(1000.0)
        assert math.isfinite(calibrated_large)
        assert calibrated_large > 0.99
        
        # Very large negative
        calibrated_neg = calibrator.calibrate(-1000.0)
        assert math.isfinite(calibrated_neg)
        assert calibrated_neg < 0.01


class TestConfidenceCalibrationPhase:
    """Test ConfidenceCalibrationPhase integration."""
    
    def test_phase_disabled_via_flag(self):
        """Phase should skip when disabled via flag."""
        from unittest.mock import MagicMock
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.confidence_calibration_phase import (
            ConfidenceCalibrationPhase
        )
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import (
            PipelineContext
        )
        
        phase = ConfidenceCalibrationPhase()
        
        mock_orch = MagicMock()
        mock_timer = MagicMock()
        
        ctx = PipelineContext(
            orchestrator=mock_orch,
            values=[1.0, 2.0, 3.0],
            timestamps=[1.0, 2.0, 3.0],
            series_id="test",
            timer=mock_timer,
            flags={"ML_ENABLE_CONFIDENCE_CALIBRATION": False},
            fused_confidence=2.0,
        )
        
        result = phase.execute(ctx)
        
        # Confidence should remain unchanged
        assert result.fused_confidence == 2.0
    
    def test_phase_calibrates_when_enabled(self):
        """Phase should calibrate confidence when enabled."""
        from unittest.mock import MagicMock
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.confidence_calibration_phase import (
            ConfidenceCalibrationPhase
        )
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import (
            PipelineContext
        )
        
        phase = ConfidenceCalibrationPhase()
        
        mock_orch = MagicMock()
        mock_timer = MagicMock()
        
        ctx = PipelineContext(
            orchestrator=mock_orch,
            values=[1.0, 2.0, 3.0],
            timestamps=[1.0, 2.0, 3.0],
            series_id="test",
            timer=mock_timer,
            flags={
                "ML_ENABLE_CONFIDENCE_CALIBRATION": True,
                "ML_CONFIDENCE_TEMPERATURE": 1.5,
            },
            fused_confidence=2.0,
        )
        
        result = phase.execute(ctx)
        
        # Confidence should be calibrated (different from raw)
        assert result.fused_confidence != 2.0
        # Should be in [0, 1]
        assert 0.0 <= result.fused_confidence <= 1.0
    
    def test_phase_handles_none_confidence(self):
        """Phase should skip when confidence is None."""
        from unittest.mock import MagicMock
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.confidence_calibration_phase import (
            ConfidenceCalibrationPhase
        )
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import (
            PipelineContext
        )
        
        phase = ConfidenceCalibrationPhase()
        
        mock_orch = MagicMock()
        mock_timer = MagicMock()
        
        ctx = PipelineContext(
            orchestrator=mock_orch,
            values=[1.0, 2.0, 3.0],
            timestamps=[1.0, 2.0, 3.0],
            series_id="test",
            timer=mock_timer,
            flags={"ML_ENABLE_CONFIDENCE_CALIBRATION": True},
            fused_confidence=None,
        )
        
        result = phase.execute(ctx)
        
        # Should return context unchanged
        assert result.fused_confidence is None
