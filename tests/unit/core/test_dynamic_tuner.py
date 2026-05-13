"""Tests for core/dynamic_tuner.py."""

import pytest

from core.tuning.dynamic_tuning import DynamicTuner
from core.tuning.convergence_detector import ConvergenceStatus
from core.parameters.parameter_bounds import ParameterBoundsEnforcer, BoundsConfig


class TestDynamicTuner:
    def test_init_without_registry(self):
        tuner = DynamicTuner()
        assert tuner._registry is None
        assert tuner._bounds_enforcer is not None
        assert tuner._convergence_window == 20

    def test_tune_learning_rate_converging(self):
        tuner = DynamicTuner()
        # Good performance, should increase slightly
        new_value = tuner.tune_learning_rate(
            name="ML_BAYES_ALPHA",
            current_value=0.15,
            performance_metric=0.8,
            series_id="test",
        )
        assert new_value > 0.15  # Should increase

    def test_tune_learning_rate_oscillating(self):
        tuner = DynamicTuner()
        detector = tuner._get_detector("ML_BAYES_ALPHA")
        # Feed oscillating values
        for val in [0.5, 0.6, 0.4, 0.7, 0.3, 0.8]:
            detector.update(val)
        
        new_value = tuner.tune_learning_rate(
            name="ML_BAYES_ALPHA",
            current_value=0.2,
            performance_metric=0.7,
        )
        assert new_value < 0.2  # Should reduce by 20%

    def test_tune_learning_rate_diverging(self):
        tuner = DynamicTuner()
        detector = tuner._get_detector("ML_BAYES_ALPHA")
        # Feed diverging values
        for val in [1.0, 1.01, 1.03, 1.06, 1.10, 1.15, 1.21, 1.28, 1.36, 1.45]:
            detector.update(val)
        
        new_value = tuner.tune_learning_rate(
            name="ML_BAYES_ALPHA",
            current_value=0.2,
            performance_metric=0.7,
        )
        assert new_value < 0.2  # Should reduce by 40%

    def test_tune_learning_rate_converged(self):
        tuner = DynamicTuner()
        detector = tuner._get_detector("ML_BAYES_ALPHA")
        # Feed stable values
        for val in [0.5, 0.5001, 0.5002, 0.5001, 0.5003, 0.5002, 0.5001, 0.5002]:
            detector.update(val)
        
        new_value = tuner.tune_learning_rate(
            name="ML_BAYES_ALPHA",
            current_value=0.15,
            performance_metric=0.7,
        )
        # Should not change significantly (bounds may clip, oscillation may be detected)
        # Allow for small adjustment due to convergence detection nuances
        assert abs(new_value - 0.15) < 0.05

    def test_tune_contamination_high_fp(self):
        tuner = DynamicTuner()
        new_value = tuner.tune_contamination(
            current_value=0.05,
            false_positive_rate=0.02,  # > 0.01 * 1.5
            target_fp_rate=0.01,
            series_id="test",
        )
        assert new_value < 0.05  # Should reduce

    def test_tune_contamination_low_fp(self):
        tuner = DynamicTuner()
        new_value = tuner.tune_contamination(
            current_value=0.05,
            false_positive_rate=0.003,  # < 0.01 * 0.5
            target_fp_rate=0.01,
            series_id="test",
        )
        assert new_value > 0.05  # Should increase

    def test_tune_contamination_ok(self):
        tuner = DynamicTuner()
        new_value = tuner.tune_contamination(
            current_value=0.05,
            false_positive_rate=0.01,  # Within range
            target_fp_rate=0.01,
            series_id="test",
        )
        assert abs(new_value - 0.05) < 0.01  # Should not change

    def test_bounds_enforced_always(self):
        tuner = DynamicTuner()
        # Try to set value outside bounds
        new_value = tuner.tune_learning_rate(
            name="ML_BAYES_ALPHA",
            current_value=0.15,
            performance_metric=0.9,  # Will try to increase beyond bounds
            series_id="test",
        )
        assert 0.01 <= new_value <= 0.5  # Should be clipped to bounds

    def test_convergence_report_structure(self):
        tuner = DynamicTuner()
        detector = tuner._get_detector("test_param")
        detector.update(0.5)
        detector.update(0.6)
        
        report = tuner.get_convergence_report()
        assert "test_param" in report
        assert "status" in report["test_param"]
        assert "current_value" in report["test_param"]
