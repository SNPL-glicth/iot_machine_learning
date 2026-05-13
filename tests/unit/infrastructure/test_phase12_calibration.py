"""Tests for Phase 12: Final Threshold Calibration."""

import pytest

from core.parameters.numerical_constants import INHIBITION_THRESHOLDS, STAT_THRESHOLDS
from infrastructure.ml.cognitive.inhibition.gate import InhibitionConfig
from ml_service.config.ml_config import OnlineBehaviorConfig
from infrastructure.ml.cognitive.inhibition.adaptive_config import AdaptiveInhibitionConfig


class TestTransientZScoreUnification:
    """Tests for transient_z_score unification with STAT_THRESHOLDS."""

    def test_transient_z_score_equals_stat_threshold_upper(self):
        """OnlineBehaviorConfig.transient_z_score should equal STAT_THRESHOLDS.Z_SCORE_UPPER."""
        config = OnlineBehaviorConfig()
        assert config.transient_z_score == STAT_THRESHOLDS.Z_SCORE_UPPER


class TestInhibitionThresholdsSingleton:
    """Tests for INHIBITION_THRESHOLDS singleton."""

    def test_inhibition_thresholds_exists_in_numerical_constants(self):
        """INHIBITION_THRESHOLDS should exist in numerical_constants."""
        assert INHIBITION_THRESHOLDS is not None

    def test_inhibition_thresholds_stability_value(self):
        """INHIBITION_THRESHOLDS.STABILITY should be 0.6."""
        assert INHIBITION_THRESHOLDS.STABILITY == 0.6

    def test_inhibition_thresholds_fit_error_value(self):
        """INHIBITION_THRESHOLDS.FIT_ERROR should be 5.0."""
        assert INHIBITION_THRESHOLDS.FIT_ERROR == 5.0

    def test_inhibition_thresholds_recent_error_value(self):
        """INHIBITION_THRESHOLDS.RECENT_ERROR should be 10.0."""
        assert INHIBITION_THRESHOLDS.RECENT_ERROR == 10.0


class TestInhibitionConfigUsesSingleton:
    """Tests for InhibitionConfig using INHIBITION_THRESHOLDS."""

    def test_inhibition_config_uses_inhibition_thresholds(self):
        """InhibitionConfig should use INHIBITION_THRESHOLDS for all three thresholds."""
        config = InhibitionConfig()
        assert config.stability_threshold == INHIBITION_THRESHOLDS.STABILITY
        assert config.fit_error_threshold == INHIBITION_THRESHOLDS.FIT_ERROR
        assert config.recent_error_threshold == INHIBITION_THRESHOLDS.RECENT_ERROR

    def test_inhibition_config_stability_from_singleton(self):
        """InhibitionConfig.stability_threshold should equal INHIBITION_THRESHOLDS.STABILITY."""
        config = InhibitionConfig()
        assert config.stability_threshold == INHIBITION_THRESHOLDS.STABILITY

    def test_inhibition_config_fit_error_from_singleton(self):
        """InhibitionConfig.fit_error_threshold should equal INHIBITION_THRESHOLDS.FIT_ERROR."""
        config = InhibitionConfig()
        assert config.fit_error_threshold == INHIBITION_THRESHOLDS.FIT_ERROR

    def test_inhibition_config_recent_error_from_singleton(self):
        """InhibitionConfig.recent_error_threshold should equal INHIBITION_THRESHOLDS.RECENT_ERROR."""
        config = InhibitionConfig()
        assert config.recent_error_threshold == INHIBITION_THRESHOLDS.RECENT_ERROR


class TestAdaptiveInhibitionConfigCompatibility:
    """Tests for AdaptiveInhibitionConfig inheritance compatibility."""

    def test_adaptive_inhibition_config_still_inherits_correctly(self):
        """AdaptiveInhibitionConfig should still inherit InhibitionConfig correctly."""
        config = AdaptiveInhibitionConfig()
        # Should inherit the base thresholds from InhibitionConfig
        assert config.stability_threshold == INHIBITION_THRESHOLDS.STABILITY
        assert config.fit_error_threshold == INHIBITION_THRESHOLDS.FIT_ERROR
        assert config.recent_error_threshold == INHIBITION_THRESHOLDS.RECENT_ERROR


class TestErrorDriftDetectorEpsilon:
    """Tests for error_drift_detector.py using EPSILON.DIVISION."""

    def test_error_drift_detector_uses_epsilon_division(self):
        """error_drift_detector.py should not contain hardcoded 1e-12 literal."""
        with open(
            "/home/nicolas/Documentos/Iot_System/iot_machine_learning/infrastructure/ml/cognitive/drift/error_drift_detector.py",
            "r"
        ) as f:
            content = f.read()
        # Verify EPSILON.DIVISION is used instead of hardcoded 1e-12
        assert "EPSILON.DIVISION" in content
        # Verify no hardcoded 1e-12 remains
        assert "1e-12" not in content
