"""Tests for Fase 10: Consolidación de Brechas Infrastructure.

Verifies that hardcoded values in infrastructure/ are replaced with
singleton references from core/parameters/.

Tests:
- Epsilon Kalman additions
- Bayesian tracker epsilon usage
- Contamination unification
- Confidence floor unification
- Regression test: no behavior change
"""

import pytest

from core.parameters.numerical_constants import CONFIDENCE, EPSILON, STAT_THRESHOLDS
from infrastructure.ml.anomaly.core.config import AnomalyDetectorConfig
from infrastructure.ml.cognitive.bayesian_weight_tracker.base import BayesianWeightTracker
from infrastructure.ml.cognitive.bayesian_weight_tracker.bayesian_weight_config import BayesianWeightConfig
from infrastructure.ml.filters.kalman_math import MIN_P, MIN_R, initialize_state, kalman_update
from infrastructure.ml.filters.kalman_math import KalmanState
from ml_service.config.decision_config import DecisionConfig
from ml_service.config.ml_config import AnomalyConfig, RegressionConfig


class TestEpsilonKalman:
    """Tests for EPSILON.KALMAN_R and EPSILON.KALMAN_P additions."""

    def test_epsilon_has_kalman_r(self):
        """EPSILON.KALMAN_R == 1e-6."""
        assert hasattr(EPSILON, "KALMAN_R")
        assert EPSILON.KALMAN_R == 1e-6

    def test_epsilon_has_kalman_p(self):
        """EPSILON.KALMAN_P == 1e-10."""
        assert hasattr(EPSILON, "KALMAN_P")
        assert EPSILON.KALMAN_P == 1e-10


class TestKalmanMathEpsilonUsage:
    """Tests for kalman_math.py using EPSILON singleton."""

    def test_kalman_math_uses_epsilon(self):
        """MIN_R is EPSILON.KALMAN_R (mismo objeto)."""
        assert MIN_R is EPSILON.KALMAN_R
        assert MIN_P is EPSILON.KALMAN_P

    def test_kalman_math_values_unchanged(self):
        """Valores siguen siendo 1e-6, 1e-10."""
        assert MIN_R == 1e-6
        assert MIN_P == 1e-10


class TestBayesianTrackerEpsilonUsage:
    """Tests for bayesian_weight_tracker/base.py using EPSILON."""

    def test_bayesian_tracker_uses_epsilon_comparison(self):
        """No usa 1e-9 literal, usa EPSILON.COMPARISON."""
        # Verify the import exists
        from infrastructure.ml.cognitive.bayesian_weight_tracker import base
        
        # Check that EPSILON is imported
        assert hasattr(base, "EPSILON")
        
        # Verify by checking the source code doesn't contain 1e-9 literal
        # This is a meta-test to ensure the refactoring was done
        import inspect
        source = inspect.getsource(base.BayesianWeightTracker._check_convergence)
        assert "1e-9" not in source
        assert "EPSILON.COMPARISON" in source


class TestContaminationUnification:
    """Tests for contamination unification across configs."""

    def test_anomaly_detector_config_contamination(self):
        """AnomalyDetectorConfig.contamination == STAT_THRESHOLDS.CONTAMINATION_DEFAULT."""
        config = AnomalyDetectorConfig()
        assert config.contamination == STAT_THRESHOLDS.CONTAMINATION_DEFAULT
        assert config.contamination == 0.005

    def test_anomaly_config_contamination(self):
        """AnomalyConfig.contamination == STAT_THRESHOLDS.CONTAMINATION_DEFAULT."""
        config = AnomalyConfig()
        assert config.contamination == STAT_THRESHOLDS.CONTAMINATION_DEFAULT
        assert config.contamination == 0.005

    def test_contamination_single_source_of_truth(self):
        """Los 2 configs tienen mismo valor."""
        anomaly_detector_config = AnomalyDetectorConfig()
        anomaly_config = AnomalyConfig()
        assert anomaly_detector_config.contamination == anomaly_config.contamination
        assert anomaly_detector_config.contamination == STAT_THRESHOLDS.CONTAMINATION_DEFAULT


class TestConfidenceFloorUnification:
    """Tests for confidence floor unification across configs."""

    def test_regression_config_min_confidence(self):
        """RegressionConfig.min_confidence == CONFIDENCE.MIN_CONFIDENCE."""
        config = RegressionConfig()
        assert config.min_confidence == CONFIDENCE.MIN_CONFIDENCE
        assert config.min_confidence == 0.3

    def test_decision_config_floor(self):
        """DecisionConfig.ML_DECISION_CONFIDENCE_FLOOR == CONFIDENCE.MIN_CONFIDENCE."""
        config = DecisionConfig()
        assert config.ML_DECISION_CONFIDENCE_FLOOR == CONFIDENCE.MIN_CONFIDENCE
        assert config.ML_DECISION_CONFIDENCE_FLOOR == 0.3

    def test_confidence_floor_single_source_of_truth(self):
        """Los 3 configs tienen mismo valor."""
        regression_config = RegressionConfig()
        decision_config = DecisionConfig()
        assert regression_config.min_confidence == decision_config.ML_DECISION_CONFIDENCE_FLOOR
        assert regression_config.min_confidence == CONFIDENCE.MIN_CONFIDENCE
        assert decision_config.ML_DECISION_CONFIDENCE_FLOOR == CONFIDENCE.MIN_CONFIDENCE


class TestRegressionTestNoBehaviorChange:
    """Regression tests to ensure no behavior change."""

    def test_kalman_filter_still_works(self):
        """Instanciar KalmanFilter, predecir."""
        # Create warmup buffer
        warmup_values = [10.0, 10.5, 11.0, 10.8, 11.2, 10.9, 11.1, 10.7, 11.0, 10.6]
        
        # Initialize state
        state = initialize_state(warmup_values, Q=1e-5)
        
        # Verify state is initialized
        assert state.initialized is True
        assert state.x_hat is not None
        assert state.P > 0
        assert state.R > 0
        
        # Apply update
        measurement = 11.0
        filtered = kalman_update(state, measurement)
        
        # Verify update worked
        assert filtered is not None
        assert isinstance(filtered, float)
        assert state.x_hat == filtered

    def test_bayesian_weight_tracker_initializes(self):
        """BayesianWeightTracker inicializa correctamente."""
        config = BayesianWeightConfig()
        tracker = BayesianWeightTracker(config=config)
        
        assert tracker is not None
        assert tracker._config is not None
        assert tracker._config.alpha == 0.15
