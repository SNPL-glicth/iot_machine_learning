"""Tests for Fase 4: per-sensor learning and drift detection."""

from __future__ import annotations

from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.per_sensor_key import (
    build_regime_key,
    build_fallback_key,
    should_use_per_sensor,
)
from iot_machine_learning.infrastructure.ml.moe.events.prediction_drift_detector import (
    DriftAlert,
    PredictionDriftDetector,
)


class TestPerSensorKey:
    def test_build_regime_key_with_series_id(self):
        assert build_regime_key("default", "stable", "sensor_01") == "default:sensor_01:stable"

    def test_build_regime_key_without_series_id(self):
        assert build_regime_key("default", "stable", None) == "default:stable"

    def test_should_use_per_sensor_no_history(self):
        assert not should_use_per_sensor("sensor_01", {}, "default", "stable")

    def test_should_use_per_sensor_insufficient_history(self):
        data = {"default:sensor_01:stable": {"kalman": 0.8}}
        assert not should_use_per_sensor("sensor_01", data, "default", "stable")

    def test_should_use_per_sensor_sufficient_history(self):
        data = {"default:sensor_01:stable": {"kalman": 0.8, "taylor": 0.7}}
        assert should_use_per_sensor("sensor_01", data, "default", "stable", threshold=1)

    def test_should_use_per_sensor_unknown_series_id(self):
        assert not should_use_per_sensor("unknown", {}, "default", "stable")


class TestPredictionDriftDetector:
    def test_no_alert_insufficient_data(self):
        detector = PredictionDriftDetector(rolling_window=10)
        for i in range(15):
            result = detector.record_error("s1", 1.0)
        assert result is None

    def test_no_alert_stable(self):
        detector = PredictionDriftDetector(rolling_window=10)
        for _ in range(50):
            result = detector.record_error("s1", 1.0)
        assert result is None

    def test_alert_on_drift(self):
        detector = PredictionDriftDetector(rolling_window=10, sigma_threshold=2.0)
        for i in range(20):
            detector.record_error("s1", 1.0 + (i % 3) * 0.1)  # baseline con varianza
        alert = None
        for _ in range(10):
            alert = detector.record_error("s1", 15.0)
        assert alert is not None
        assert isinstance(alert, DriftAlert)
        assert alert.sigma_distance > 2.0

    def test_get_mae_none_without_data(self):
        detector = PredictionDriftDetector()
        assert detector.get_mae("unknown") is None

    def test_get_mae_returns_value(self):
        detector = PredictionDriftDetector(rolling_window=5)
        for _ in range(5):
            detector.record_error("s1", 2.0)
        mae = detector.get_mae("s1")
        assert mae is not None
        assert abs(mae - 2.0) < 0.01


class TestWeightsMixinFallback:
    def test_get_weights_fallback_without_history(self):
        from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.weights_mixin import WeightsMixin
        from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.bayesian_weight_config import BayesianWeightConfig

        class DummyMixin(WeightsMixin):
            def __init__(self):
                self._domain_namespace = "default"
                self._accuracy = {}
                self._config = BayesianWeightConfig()
                self._redis = type("R", (), {"get_weights": lambda *a, **k: None})()

        mixin = DummyMixin()
        weights = mixin.get_weights("stable", ["a", "b"], series_id="sensor_no_history")
        assert abs(sum(weights.values()) - 1.0) < 1e-9
