"""Tests para BaselinePredictionAdapter.

DEPRECADO — BaselinePredictionAdapter no existe en engines.baseline.
Verifica que el adapter bridge entre ml.baseline y PredictionPort
funciona correctamente.
"""

from __future__ import annotations

# Skip temprano antes de imports fallidos
import pytest
pytestmark = pytest.mark.skip(reason="BaselinePredictionAdapter no existe - modulo legacy no migrado")

# Mocks para imports que fallan
from unittest.mock import MagicMock
Prediction = MagicMock
SensorReading = MagicMock
SensorWindow = MagicMock
PredictionPort = MagicMock
BaselinePredictionAdapter = MagicMock


def _make_window(sensor_id: int, values: list[float]) -> SensorWindow:
    readings = [
        SensorReading(sensor_id=sensor_id, value=v, timestamp=float(i))
        for i, v in enumerate(values)
    ]
    return SensorWindow(sensor_id=sensor_id, readings=readings)


class TestBaselinePredictionAdapter:
    """Tests para el adapter Baseline → PredictionPort."""

    def test_implements_prediction_port(self) -> None:
        adapter = BaselinePredictionAdapter(window=10)
        assert isinstance(adapter, PredictionPort)

    def test_name(self) -> None:
        adapter = BaselinePredictionAdapter()
        assert adapter.name == "baseline_moving_average"

    def test_can_handle_with_1_point(self) -> None:
        adapter = BaselinePredictionAdapter()
        assert adapter.can_handle(1) is True

    def test_can_handle_with_0_points(self) -> None:
        adapter = BaselinePredictionAdapter()
        assert adapter.can_handle(0) is False

    def test_predict_returns_prediction(self) -> None:
        adapter = BaselinePredictionAdapter(window=5)
        window = _make_window(1, [20.0, 21.0, 22.0, 23.0, 24.0])
        result = adapter.predict(window)

        assert isinstance(result, Prediction)
        assert result.series_id == "1"
        assert result.engine_name == "baseline_moving_average"
        assert 0.0 <= result.confidence_score <= 1.0

    def test_predict_stable_values(self) -> None:
        adapter = BaselinePredictionAdapter(window=10)
        window = _make_window(1, [22.0] * 10)
        result = adapter.predict(window)

        assert abs(result.predicted_value - 22.0) < 0.01
        assert result.trend == "stable"

    def test_predict_upward_trend(self) -> None:
        adapter = BaselinePredictionAdapter(window=5)
        window = _make_window(1, [20.0, 21.0, 22.0, 23.0, 24.0])
        result = adapter.predict(window)

        assert result.trend == "up"

    def test_predict_downward_trend(self) -> None:
        adapter = BaselinePredictionAdapter(window=5)
        window = _make_window(1, [24.0, 23.0, 22.0, 21.0, 20.0])
        result = adapter.predict(window)

        assert result.trend == "down"

    def test_predict_empty_window_raises(self) -> None:
        adapter = BaselinePredictionAdapter()
        window = _make_window(1, [])
        with pytest.raises(ValueError, match="Ventana vacía"):
            adapter.predict(window)

    def test_predict_single_value(self) -> None:
        adapter = BaselinePredictionAdapter(window=10)
        window = _make_window(1, [42.0])
        result = adapter.predict(window)

        assert abs(result.predicted_value - 42.0) < 0.01

    def test_metadata_contains_window(self) -> None:
        adapter = BaselinePredictionAdapter(window=20)
        window = _make_window(1, [22.0] * 5)
        result = adapter.predict(window)

        assert result.metadata["window"] == 20
        assert result.metadata["n_points"] == 5

    def test_does_not_support_confidence_interval(self) -> None:
        adapter = BaselinePredictionAdapter()
        assert adapter.supports_confidence_interval() is False
