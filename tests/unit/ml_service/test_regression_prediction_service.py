"""Tests para RegressionPredictionService.

Verifica lógica de Modeling pura — sin BD, sin I/O.
Dependencias pesadas (sqlalchemy) mockeadas vía conftest.py.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from iot_machine_learning.ml_service.runners.common.regression_prediction_service import (
    RegressionPredictionService,
    PredictionResult,
)


# --- Fake objects para evitar dependencias reales ---

@dataclass(frozen=True)
class FakeRegressionConfig:
    window_points: int = 100
    min_confidence: float = 0.3
    max_confidence: float = 0.95
    horizon_minutes: int = 10
    min_points: int = 10
    model_type: str = "linear"


@dataclass
class FakeRegressionModel:
    sensor_id: int = 1
    coef_: float = 0.1
    intercept_: float = 20.0
    r2: float = 0.85
    horizon_minutes: int = 10


@dataclass
class FakeSensorSeries:
    sensor_id: int = 1
    timestamps: list = None
    values: list = None

    def __post_init__(self):
        if self.values is None:
            self.values = [20.0, 21.0, 22.0, 23.0, 24.0]
        if self.timestamps is None:
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            self.timestamps = [
                base + timedelta(minutes=i) for i in range(len(self.values))
            ]


class FakeIsoTrainer:
    """Fake IsolationForestTrainer."""

    def __init__(self, anomaly: bool = False, score: float = 0.0):
        self._anomaly = anomaly
        self._score = score

    def fit_for_sensor(self, sensor_id, residuals):
        return MagicMock()

    def score_new_point(self, sensor_id, last_residual):
        return self._score, self._anomaly


# --- Tests ---

class TestPredictFallback:
    """Tests para predicción fallback (promedio simple)."""

    def test_returns_prediction_result(self) -> None:
        svc = RegressionPredictionService()
        cfg = FakeRegressionConfig()
        result = svc.predict_fallback([20.0, 22.0, 24.0], cfg)

        assert isinstance(result, PredictionResult)

    def test_predicted_value_is_average_of_last_5(self) -> None:
        svc = RegressionPredictionService()
        cfg = FakeRegressionConfig()
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = svc.predict_fallback(values, cfg)

        expected = sum(values) / len(values)
        assert abs(result.predicted_value - expected) < 0.01

    def test_trend_is_stable(self) -> None:
        svc = RegressionPredictionService()
        cfg = FakeRegressionConfig()
        result = svc.predict_fallback([20.0, 20.0], cfg)

        assert result.trend == "stable"

    def test_anomaly_is_false(self) -> None:
        svc = RegressionPredictionService()
        cfg = FakeRegressionConfig()
        result = svc.predict_fallback([20.0], cfg)

        assert result.anomaly is False
        assert result.anomaly_score == 0.0

    def test_confidence_is_min(self) -> None:
        svc = RegressionPredictionService()
        cfg = FakeRegressionConfig(min_confidence=0.25)
        result = svc.predict_fallback([20.0], cfg)

        assert result.confidence == 0.25

    def test_window_points_effective(self) -> None:
        svc = RegressionPredictionService()
        cfg = FakeRegressionConfig()
        result = svc.predict_fallback([1.0, 2.0, 3.0], cfg)

        assert result.window_points_effective == 3

    def test_single_value(self) -> None:
        svc = RegressionPredictionService()
        cfg = FakeRegressionConfig()
        result = svc.predict_fallback([42.0], cfg)

        assert abs(result.predicted_value - 42.0) < 0.01

    def test_many_values_uses_last_5(self) -> None:
        svc = RegressionPredictionService()
        cfg = FakeRegressionConfig()
        values = [100.0] * 50 + [10.0, 20.0, 30.0, 40.0, 50.0]
        result = svc.predict_fallback(values, cfg)

        expected = (10.0 + 20.0 + 30.0 + 40.0 + 50.0) / 5
        assert abs(result.predicted_value - expected) < 0.01


class TestPredictWithModel:
    """Tests para predicción con modelo de regresión."""

    def test_returns_prediction_result(self) -> None:
        svc = RegressionPredictionService()
        series = FakeSensorSeries()
        model = FakeRegressionModel()
        cfg = FakeRegressionConfig()
        iso = FakeIsoTrainer()

        result = svc.predict_with_model(
            series, model, last_minutes=4.0, reg_cfg=cfg,
            iso_trainer=iso, sensor_id=1
        )

        assert isinstance(result, PredictionResult)

    def test_confidence_bounded(self) -> None:
        svc = RegressionPredictionService()
        series = FakeSensorSeries()
        model = FakeRegressionModel(r2=0.99)
        cfg = FakeRegressionConfig(min_confidence=0.1, max_confidence=0.9)
        iso = FakeIsoTrainer()

        result = svc.predict_with_model(
            series, model, last_minutes=4.0, reg_cfg=cfg,
            iso_trainer=iso, sensor_id=1
        )

        assert cfg.min_confidence <= result.confidence <= cfg.max_confidence

    def test_anomaly_detected(self) -> None:
        svc = RegressionPredictionService()
        series = FakeSensorSeries()
        model = FakeRegressionModel()
        cfg = FakeRegressionConfig()
        iso = FakeIsoTrainer(anomaly=True, score=0.85)

        result = svc.predict_with_model(
            series, model, last_minutes=4.0, reg_cfg=cfg,
            iso_trainer=iso, sensor_id=1
        )

        assert result.anomaly is True
        assert result.anomaly_score == 0.85

    def test_no_anomaly(self) -> None:
        svc = RegressionPredictionService()
        series = FakeSensorSeries()
        model = FakeRegressionModel()
        cfg = FakeRegressionConfig()
        iso = FakeIsoTrainer(anomaly=False, score=0.1)

        result = svc.predict_with_model(
            series, model, last_minutes=4.0, reg_cfg=cfg,
            iso_trainer=iso, sensor_id=1
        )

        assert result.anomaly is False

    def test_window_points_matches_series(self) -> None:
        svc = RegressionPredictionService()
        values = [20.0 + i for i in range(15)]
        series = FakeSensorSeries(values=values)
        model = FakeRegressionModel()
        cfg = FakeRegressionConfig()
        iso = FakeIsoTrainer()

        result = svc.predict_with_model(
            series, model, last_minutes=14.0, reg_cfg=cfg,
            iso_trainer=iso, sensor_id=1
        )

        assert result.window_points_effective == 15
